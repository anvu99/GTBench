"""
gamingbench/ltm/cpqa_store.py

Persistence layer for ConsolidatedPQA agent.
Each opponent has:
  - a flat stat_pool: {stat_id -> stat_obj} for all statistical trackers
  - a list of MemoryItems (topic + list of stat_id references + synthesized answer)
  - a pre-rendered consolidated_block string injected into every game prompt
  - a games_observed counter

Stat types: DISTRIBUTION and MEAN_VAR only.
"""

import json
import os
import time
import uuid
import threading
from typing import Dict, List, Optional, Any


VALID_STAT_TYPES = {"CATEGORY", "DISTRIBUTION", "MEAN_VAR"}


def _default_storage(stat_type: str) -> dict:
    if stat_type in ("CATEGORY", "DISTRIBUTION"):
        return {"buckets": {}, "total": 0}
    elif stat_type == "MEAN_VAR":
        return {"n": 0, "sum": 0.0, "sum_sq": 0.0}
    return {}


def _format_stat_value(stat: dict) -> str:
    """Renders a stat's storage as a human-readable string for prompt injection."""
    stype = stat.get("type", "")
    st = stat.get("storage", {})
    if stype in ("CATEGORY", "DISTRIBUTION"):
        b = st.get("buckets", {})
        total = st.get("total", 0)
        if not b:
            return "No data yet."
        parts = []
        if total == 0:
            # For CATEGORY, show all labels even at 0 so Step 6 knows what's valid
            for k in sorted(b.keys()):
                parts.append(f"{k}: 0/{total} (0%)")
        else:
            for k, v in sorted(b.items(), key=lambda x: -x[1]):
                pct = (v / total * 100) if total else 0
                parts.append(f"{k}: {v}/{total} ({pct:.0f}%)")
        return ", ".join(parts)
    elif stype == "MEAN_VAR":
        n = st.get("n", 0)
        if n == 0:
            return "No data yet."
        import math
        mean = st["sum"] / n
        var = max(0.0, (st["sum_sq"] / n) - (mean ** 2))
        std = math.sqrt(var)
        return f"mean={mean:.3f}, std={std:.3f} (n={n})"
    return "Unknown type."


def _stat_update_count(stat: dict) -> int:
    """Returns the total number of observations recorded for a stat."""
    stype = stat.get("type", "")
    st = stat.get("storage", {})
    if stype == "MEAN_VAR":
        return st.get("n", 0)
    elif stype in ("CATEGORY", "DISTRIBUTION"):
        return st.get("total", 0)
    return 0


class CPQAStore:
    """
    Stores a flat stat pool and memory items per opponent.

    Schema:
    {
      opponent_key: {
        "stat_pool": {
          "stat_xxxx": {
            "type": "MEAN_VAR",
            "description": "...",
            "pseudocode": "...",
            "storage": { "n": 0, "sum": 0.0, "sum_sq": 0.0 }
          },
          "stat_yyyy": {
            "type": "DISTRIBUTION",
            "description": "...",
            "pseudocode": "...",
            "storage": { "buckets": {}, "total": 0 }
          }
        },
        "memories": [
          {
            "id": "mem_xxxx",
            "topic": "Does opponent underbid?",
            "stat_ids": ["stat_xxxx", "stat_yyyy"],
            "answer": "...",
            "games_observed": 0,
            "created_at": float,
            "updated_at": float
          }
        ],
        "consolidated_block": "=== OPPONENT PROFILE ===\\n...",
        "games_observed": 0
      }
    }
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Opponent-level helpers
    # ------------------------------------------------------------------

    def _ensure_opponent(self, opp_key: str):
        if opp_key not in self._data:
            self._data[opp_key] = {
                "stat_pool": {},
                "memories": [],
                "consolidated_block": "",
                "games_observed": 0,
            }

    def increment_games_observed(self, opp_key: str):
        self._ensure_opponent(opp_key)
        self._data[opp_key]["games_observed"] += 1

    def get_games_observed(self, opp_key: str) -> int:
        return self._data.get(opp_key, {}).get("games_observed", 0)

    def get_consolidated_block(self, opp_key: str) -> str:
        return self._data.get(opp_key, {}).get("consolidated_block", "")

    def set_consolidated_block(self, opp_key: str, block: str):
        self._ensure_opponent(opp_key)
        self._data[opp_key]["consolidated_block"] = block

    # ------------------------------------------------------------------
    # Stat pool CRUD
    # ------------------------------------------------------------------

    def add_stat(self, opp_key: str, stat_type: str, description: str, pseudocode: str, initial_labels: list = None) -> Optional[str]:
        """
        Creates a new stat tracker in the global pool. Returns the new UUID (stat_id).
        
        Important Type Distinction:
        - DISTRIBUTION: Exploratory. Start with empty buckets. New labels are discovered during Step 6.
        - CATEGORY: Exhaustive. The `initial_labels` list strictly defines all possible buckets.
          We pre-populate them at count 0 here. Step 6 is forbidden from inventing new labels for CATEGORY.
        """
        if stat_type not in VALID_STAT_TYPES:
            return None
        self._ensure_opponent(opp_key)
        stat_id = f"stat_{uuid.uuid4().hex[:8]}"
        storage = _default_storage(stat_type)
        if stat_type == "CATEGORY" and initial_labels:
            for label in initial_labels:
                label_str = str(label).strip()
                if label_str:
                    storage["buckets"][label_str] = 0
        self._data[opp_key]["stat_pool"][stat_id] = {
            "type": stat_type,
            "description": description,
            "pseudocode": pseudocode,
            "storage": storage,
        }
        return stat_id

    def get_stat(self, opp_key: str, stat_id: str) -> Optional[Dict[str, Any]]:
        """Returns a stat from the pool by ID."""
        return self._data.get(opp_key, {}).get("stat_pool", {}).get(stat_id)

    def get_all_stats(self, opp_key: str) -> Dict[str, Dict[str, Any]]:
        """Returns the full stat_pool dict {stat_id -> stat_obj}."""
        return self._data.get(opp_key, {}).get("stat_pool", {})

    def get_top_updated_stats(self, opp_key: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Returns up to n stats that have at least one observation (n > 0),
        sorted descending by total update count.
        Each returned dict includes the stat_id as a key.
        """
        pool = self.get_all_stats(opp_key)
        active = []
        for stat_id, stat in pool.items():
            count = _stat_update_count(stat)
            if count > 0:
                active.append({**stat, "stat_id": stat_id, "_update_count": count})
        active.sort(key=lambda s: -s["_update_count"])
        return active[:n]

    def evict_stat(self, opp_key: str, stat_id: str) -> bool:
        """
        Removes a stat from the pool and cleans up all memory references.
        Returns True if the stat was found and removed.
        """
        pool = self._data.get(opp_key, {}).get("stat_pool", {})
        if stat_id not in pool:
            return False
        del pool[stat_id]
        # Clean up references in memories
        for mem in self.get_memories(opp_key):
            if stat_id in mem.get("stat_ids", []):
                mem["stat_ids"] = [s for s in mem["stat_ids"] if s != stat_id]
        return True

    # ------------------------------------------------------------------
    # Memory item CRUD
    # ------------------------------------------------------------------

    def get_memories(self, opp_key: str) -> List[Dict[str, Any]]:
        return self._data.get(opp_key, {}).get("memories", [])

    def get_memory(self, opp_key: str, mem_id: str) -> Optional[Dict[str, Any]]:
        for m in self.get_memories(opp_key):
            if m["id"] == mem_id:
                return m
        return None

    def get_memory_stats(self, opp_key: str, mem_id: str) -> List[Dict[str, Any]]:
        """
        Resolves a memory item's stat_ids list to actual stat objects from the pool.
        Returns a list of stat dicts, each augmented with its stat_id.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return []
        pool = self.get_all_stats(opp_key)
        result = []
        for sid in mem.get("stat_ids", []):
            stat = pool.get(sid)
            if stat:
                result.append({**stat, "stat_id": sid})
        return result

    def add_memory(self, opp_key: str, topic: str, stat_ids: List[str], answer: str = "") -> str:
        """
        Creates a new memory item referencing existing stat_ids from the pool.
        Returns the new mem_id.
        """
        self._ensure_opponent(opp_key)
        mem_id = f"mem_{uuid.uuid4().hex[:8]}"
        # Validate that all stat_ids exist in the pool
        pool = self._data[opp_key]["stat_pool"]
        valid_stat_ids = [s for s in stat_ids if s in pool]
        self._data[opp_key]["memories"].append({
            "id": mem_id,
            "topic": topic,
            "stat_ids": valid_stat_ids,
            "answer": answer,
            "games_observed": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        })
        return mem_id

    def update_memory_answer(self, opp_key: str, mem_id: str, new_answer: str):
        """Replaces the answer field of a memory item."""
        mem = self.get_memory(opp_key, mem_id)
        if mem:
            mem["answer"] = new_answer
            mem["updated_at"] = time.time()

    def _get_stat_ref_count(self, opp_key: str, stat_id: str) -> int:
        """Returns how many memory items currently reference a given stat_id."""
        return sum(1 for m in self.get_memories(opp_key) if stat_id in m.get("stat_ids", []))

    def delete_memory(self, opp_key: str, mem_id: str) -> bool:
        """
        Removes a memory item (topic) from the store.
        
        Garbage Collection Rule:
        If deleting this memory leaves any of its referenced stats "orphaned" (meaning no other 
        memory points to them), we must decide whether to delete the stat from the global pool.
        We ONLY delete an orphaned stat if it has ZERO data (i.e. it was never successfully updated).
        If a stat has data, we keep it in the pool so future memories can reference it.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return False
        stat_ids_to_check = list(mem.get("stat_ids", []))
        # Remove the memory
        mems = self.get_memories(opp_key)
        self._data[opp_key]["memories"] = [m for m in mems if m["id"] != mem_id]
        # Prune orphaned zero-data stats
        pool = self._data.get(opp_key, {}).get("stat_pool", {})
        for sid in stat_ids_to_check:
            if self._get_stat_ref_count(opp_key, sid) == 0:
                stat = pool.get(sid)
                if stat and _stat_update_count(stat) == 0:
                    del pool[sid]
        return True

    def increment_memory_games(self, opp_key: str, mem_id: str):
        mem = self.get_memory(opp_key, mem_id)
        if mem:
            mem["games_observed"] = mem.get("games_observed", 0) + 1

    def evict_stat_from_memory(self, opp_key: str, mem_id: str, stat_id: str) -> bool:
        """
        Removes a single stat_id reference from a memory item's list of trackers.
        
        Garbage Collection Rule:
        Just like delete_memory, if this eviction leaves the stat orphaned AND the stat has 
        zero data, we prune it from the global pool to prevent zero-data bloat.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return False
        before = len(mem.get("stat_ids", []))
        mem["stat_ids"] = [s for s in mem.get("stat_ids", []) if s != stat_id]
        removed = len(mem["stat_ids"]) < before
        if removed and self._get_stat_ref_count(opp_key, stat_id) == 0:
            pool = self._data.get(opp_key, {}).get("stat_pool", {})
            stat = pool.get(stat_id)
            if stat and _stat_update_count(stat) == 0:
                del pool[stat_id]
        return removed

    # ------------------------------------------------------------------
    # Stat delta application
    # ------------------------------------------------------------------

    def apply_stat_deltas(self, opp_key: str, stat_id: str, deltas: dict) -> bool:
        """
        Applies typed delta increments to a global stat's storage block.
        
        - DISTRIBUTION and CATEGORY: deltas["buckets"] = {label: count, ...}
          Both types just accumulate incoming counts into their buckets and increment "total".
          The difference is handled upstream: Step 6b ensures DISTRIBUTION labels are canonicalized,
          and Step 6 refuses to generate new labels for CATEGORY.
        - MEAN_VAR: deltas["n"], deltas["sum"], deltas["sum_sq"]
        
        Thread-safe via self._lock.
        """
        with self._lock:
            stat = self.get_stat(opp_key, stat_id)
            if not stat:
                return False
            stype = stat["type"]
            storage = stat["storage"]

            def _safe_int(v):
                try: return int(v)
                except (TypeError, ValueError): return 0

            def _safe_float(v):
                try: return float(v)
                except (TypeError, ValueError): return 0.0

            if stype in ("CATEGORY", "DISTRIBUTION"):
                bucket_deltas = deltas.get("buckets", {})
                for label, count in bucket_deltas.items():
                    label_str = str(label)
                    storage["buckets"][label_str] = storage["buckets"].get(label_str, 0) + _safe_int(count)
                    storage["total"] = storage.get("total", 0) + _safe_int(count)
            elif stype == "MEAN_VAR":
                storage["n"] += _safe_int(deltas.get("n", 0))
                storage["sum"] += _safe_float(deltas.get("sum", 0.0))
                storage["sum_sq"] += _safe_float(deltas.get("sum_sq", 0.0))

        return True

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_stats_for_prompt(self, opp_key: str, mem_id: str) -> str:
        """Returns a human-readable block of all stats for a memory item."""
        stats = self.get_memory_stats(opp_key, mem_id)
        if not stats:
            return "No statistical trackers yet."
        lines = []
        for s in stats:
            val_str = _format_stat_value(s)
            lines.append(
                f"  [{s['stat_id']}] {s['type']}: {s['description']}\n"
                f"    Pseudocode: {s['pseudocode']}\n"
                f"    Current values: {val_str}"
            )
        return "\n".join(lines)

    def format_all_stats_for_update(self, opp_key: str, chunk_size: int = 15) -> List[str]:
        """
        Formats all stats in the pool for the stat update LLM call.
        Returns a list of formatted text blocks, each containing up to chunk_size stats.
        """
        pool = self.get_all_stats(opp_key)
        stat_blocks = []
        current_chunk_lines = []
        count = 0

        for stat_id, stat in pool.items():
            val_str = _format_stat_value(stat)
            current_chunk_lines.append(
                f"[stat_id={stat_id}] {stat['type']}: {stat['description']}\n"
                f"  Pseudocode: {stat['pseudocode']}\n"
                f"  Current values: {val_str}"
            )
            count += 1
            if count >= chunk_size:
                stat_blocks.append("\n\n".join(current_chunk_lines))
                current_chunk_lines = []
                count = 0

        if current_chunk_lines:
            stat_blocks.append("\n\n".join(current_chunk_lines))

        return stat_blocks

    def format_top_stats_for_reference(self, opp_key: str, n: int = 10) -> str:
        """
        Returns a formatted block of the top-N most-updated stats for use
        as reference context in the Step 5b prompt.
        """
        top_stats = self.get_top_updated_stats(opp_key, n=n)
        if not top_stats:
            return "No existing stats with data yet."
        lines = []
        for s in top_stats:
            val_str = _format_stat_value(s)
            lines.append(
                f"[stat_id={s['stat_id']}] {s['type']}: {s['description']}\n"
                f"  Pseudocode: {s['pseudocode']}\n"
                f"  Current values: {val_str}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._data = loaded
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: CPQAStore failed to load {filepath}: {e}. Starting fresh.")
            self._data = {}
