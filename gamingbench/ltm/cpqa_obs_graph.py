"""
gamingbench/ltm/cpqa_obs_graph.py

Observation Graph Store for CPQA v2 (simplified).

Replaces the old node-graph design. Per-opponent schema:
  - memories: synthesized memory entries (name, description, anchor, bullets, embedding)
  - consolidated_block: rendered profile injected into every game prompt
  - games_observed: total games seen against this opponent

FAISS indexes are built directly on memory anchor embeddings — no intermediate
observation_node layer. Retrieval is a single-hop: query → mem_id → memory.

Three observation types (UNCONDITIONAL / CONDITIONAL / INFERENCE) are tracked
via the memory's "type" field. Each type has its own FAISS sub-index.
"""

import json
import os
import time
import uuid
import threading
import numpy as np
from typing import Dict, List, Optional, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

OBS_TYPES = {"UNCONDITIONAL", "CONDITIONAL", "INFERENCE"}


def _make_mem_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Anchor-field helpers (used externally by the agent for embedding new obs)
# ---------------------------------------------------------------------------

def _obs_anchor_text(node: dict) -> str:
    """Returns the anchor field text of a raw observation dict (by type)."""
    t = node.get("type", "")
    if t == "UNCONDITIONAL":
        return node.get("behavior", "")
    elif t == "CONDITIONAL":
        return node.get("game_state", "")
    elif t == "INFERENCE":
        return node.get("public_action", "")
    return ""


def _format_memory_for_prompt(mem: dict) -> str:
    """Formats a memory entry for injection into prompts."""
    lines = [
        f"Memory ID: {mem['id']} | Type: {mem['type']}",
        f"Name: {mem.get('name', '')}",
        f"Description: {mem.get('description', '')}",
    ]
    anchor = mem.get("anchor", "")
    if anchor:
        label = "Game State" if mem["type"] == "CONDITIONAL" else "Public Action"
        if mem["type"] == "UNCONDITIONAL":
            label = "Anchor"
        lines.append(f"Anchor ({label}): {anchor}")

    bullets = mem.get("bullets", [])
    if bullets:
        t = mem.get("type", "")
        if t == "UNCONDITIONAL":
            bullet_label = "Observed Behavioral Variants"
        elif t == "CONDITIONAL":
            bullet_label = "Observed Reactions (given the trigger above)"
        else:  # INFERENCE
            bullet_label = "Possible Hidden Drivers (inferred from the public action above)"
        lines.append(f"{bullet_label}:")
        for b in bullets:
            lines.append(f"  [{b['id']}] {b['description']} — count: {b['count']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ObsGraphStore
# ---------------------------------------------------------------------------

class ObsGraphStore:
    """
    Simplified observation store for CPQA v2.

    Thread-safe. Schema per opponent:
      - memories:           dict[mem_id -> memory_entry]
      - consolidated_block: str
      - games_observed:     int

    FAISS is built on memory anchor embeddings (one sub-index per obs type).
    There are no intermediate observation_nodes.
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Live FAISS indexes (never persisted; rebuilt from memory embeddings on load)
        # Structure: opp_key -> obs_type -> {"index": faiss_index, "mem_ids": [...]}
        self._faiss_indexes: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Opponent scaffolding
    # ------------------------------------------------------------------

    def _ensure_opponent(self, opp_key: str):
        if opp_key not in self._data:
            self._data[opp_key] = {
                "memories": {},
                "consolidated_block": "",
                "games_observed": 0,
            }
        if opp_key not in self._faiss_indexes:
            self._faiss_indexes[opp_key] = {
                "UNCONDITIONAL": None,
                "CONDITIONAL": None,
                "INFERENCE": None,
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
    # FAISS index management (on memory anchor embeddings)
    # ------------------------------------------------------------------

    def _rebuild_faiss_index(self, opp_key: str, obs_type: str):
        """
        Rebuilds the FAISS flat index for a given obs_type using the anchor
        embeddings stored on memory entries of that type.
        """
        if not FAISS_AVAILABLE:
            return
        mems = self._data.get(opp_key, {}).get("memories", {})
        entries = [
            (mem_id, mem["embedding"])
            for mem_id, mem in mems.items()
            if mem.get("type") == obs_type and mem.get("embedding") is not None
        ]
        if not entries:
            with self._lock:
                self._faiss_indexes[opp_key][obs_type] = None
            return

        mem_ids = [e[0] for e in entries]
        vecs = np.array([e[1] for e in entries], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms

        dim = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        with self._lock:
            self._faiss_indexes[opp_key][obs_type] = {
                "index": index,
                "mem_ids": mem_ids,
            }

    def query_similar_memories(self, opp_key: str, obs_type: str,
                               query_vec: np.ndarray, top_k: int = 5,
                               threshold: float = 0.0) -> List[dict]:
        """
        Returns up to top_k existing memory entries whose anchor embedding is
        most similar to query_vec (cosine similarity, descending).
        Only memories with score >= threshold are returned.

        Returns list of {"mem_id": str, "score": float, "memory": dict}.
        """
        self._ensure_opponent(opp_key)
        with self._lock:
            idx_data = self._faiss_indexes.get(opp_key, {}).get(obs_type)

        if idx_data is None or not FAISS_AVAILABLE:
            return []

        index = idx_data["index"]
        mem_ids = idx_data["mem_ids"]

        q = np.array(query_vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        k = min(top_k, index.ntotal)
        if k == 0:
            return []

        scores, indices = index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            mem_id = mem_ids[idx]
            mem = self.get_memory(opp_key, mem_id)
            if mem:
                results.append({"mem_id": mem_id, "score": float(score), "memory": mem})
        return results

    # ------------------------------------------------------------------
    # Memory entry CRUD
    # ------------------------------------------------------------------

    def get_memory(self, opp_key: str, mem_id: str) -> Optional[dict]:
        return self._data.get(opp_key, {}).get("memories", {}).get(mem_id)

    def get_all_memories(self, opp_key: str) -> List[dict]:
        return list(self._data.get(opp_key, {}).get("memories", {}).values())

    def upsert_memory(self, opp_key: str, mem_id: Optional[str], mem_type: str,
                      name: str, description: str, anchor: str,
                      bullets: List[dict],
                      embedding: Optional[np.ndarray] = None,
                      rebuild_faiss: bool = True) -> str:
        """
        Creates or fully replaces a memory entry. Returns mem_id.

        embedding: anchor-field embedding (np.ndarray or list). If provided,
                   the FAISS index for mem_type is rebuilt after upsert.

        rebuild_faiss: if True (default), immediately rebuilds the FAISS index
                       for mem_type. Set to False when calling from concurrent
                       threads (e.g. Step B) to avoid rebuild races — caller is
                       then responsible for rebuilding once after all writes.

        bullets: list of {"id": str, "description": str, "count": int}
        """
        self._ensure_opponent(opp_key)
        if mem_id is None:
            mem_id = _make_mem_id()

        emb_list = None
        if embedding is not None:
            emb_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        with self._lock:
            self._data[opp_key]["memories"][mem_id] = {
                "id": mem_id,
                "type": mem_type,
                "name": name,
                "description": description,
                "anchor": anchor,
                "bullets": bullets,
                "embedding": emb_list,
                "updated_at": time.time(),
            }

        # Rebuild FAISS for this type (outside lock to avoid holding it during rebuild).
        # Skipped when rebuild_faiss=False — caller must call _rebuild_faiss_index() after
        # all concurrent upserts complete.
        if rebuild_faiss:
            self._rebuild_faiss_index(opp_key, mem_type)
        return mem_id

    def increment_bullet_count(self, opp_key: str, mem_id: str,
                               bullet_id: str, delta: int = 1):
        """Increments the count of a specific bullet in a memory entry."""
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return
        with self._lock:
            for b in mem.get("bullets", []):
                if b["id"] == bullet_id:
                    b["count"] = b.get("count", 0) + delta
                    mem["updated_at"] = time.time()
                    return

    def add_bullet_to_memory(self, opp_key: str, mem_id: str,
                             description: str) -> Optional[str]:
        """
        Appends a new bullet to a memory entry. Auto-assigns next letter ID.
        Returns the new bullet ID, or None if memory not found.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return None
        with self._lock:
            bullets = mem.get("bullets", [])
            next_id = chr(ord("A") + len(bullets))
            bullets.append({"id": next_id, "description": description, "count": 1})
            mem["bullets"] = bullets
            mem["updated_at"] = time.time()
        return next_id

    def delete_memory(self, opp_key: str, mem_id: str):
        """
        Deletes a memory entry and rebuilds the FAISS index for its type.
        Used by Step D to discard duplicate components after merging.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return
        mem_type = mem.get("type", "UNCONDITIONAL")
        with self._lock:
            self._data.get(opp_key, {}).get("memories", {}).pop(mem_id, None)
        self._rebuild_faiss_index(opp_key, mem_type)

    # ------------------------------------------------------------------
    # Formatting helpers for prompts
    # ------------------------------------------------------------------

    def format_memories_for_confirm(self, opp_key: str) -> str:
        """
        Renders all memory entries for the post-game confirmation prompt.
        Returns a formatted multi-line string.
        """
        mems = self.get_all_memories(opp_key)
        if not mems:
            return "(No memories yet.)"
        blocks = [_format_memory_for_prompt(m) for m in mems]
        return "\n\n---\n\n".join(blocks)

    def format_memories_by_type_for_dedup(self, opp_key: str, obs_type: str) -> str:
        """
        Renders all memory entries of a given type for the Step D dedup prompt.
        Returns '(No memories of this type yet.)' if none exist.
        """
        mems = [m for m in self.get_all_memories(opp_key) if m.get("type") == obs_type]
        if not mems:
            return "(No memories of this type yet.)"
        blocks = [_format_memory_for_prompt(m) for m in mems]
        return "\n\n---\n\n".join(blocks)

    def format_raw_obs_for_synthesis(self, raw_obs_list: List[dict]) -> str:
        """
        Formats a list of raw observation dicts for the Step B synthesis prompt.
        Replaces the old format_component_for_synthesis (which read from obs nodes).
        """
        lines = []
        for obs in raw_obs_list:
            obs_type = obs.get("type", "")
            count = obs.get("count", 1)
            if obs_type == "UNCONDITIONAL":
                lines.append(f"  - [count: {count}] {obs.get('behavior', '')}")
            elif obs_type == "CONDITIONAL":
                lines.append(
                    f"  - [count: {count}] Trigger: {obs.get('game_state', '')} | "
                    f"Action: {obs.get('action', '')}"
                )
            elif obs_type == "INFERENCE":
                lines.append(
                    f"  - [count: {count}] Public Action: {obs.get('public_action', '')} | "
                    f"Underlying Driver: {obs.get('underlying_driver', '')}"
                )
        return "\n".join(lines) if lines else "(empty)"

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
            if not isinstance(loaded, dict):
                return
            # Support old format (with observation_nodes): just ignore those keys.
            # Extract only what we need.
            clean = {}
            for opp_key, opp_data in loaded.items():
                clean[opp_key] = {
                    "memories": opp_data.get("memories", {}),
                    "consolidated_block": opp_data.get("consolidated_block", ""),
                    "games_observed": opp_data.get("games_observed", 0),
                }
            self._data = clean
            # Rebuild FAISS indexes from stored memory embeddings
            for opp_key in self._data:
                self._ensure_opponent(opp_key)
                for obs_type in OBS_TYPES:
                    self._rebuild_faiss_index(opp_key, obs_type)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: ObsGraphStore failed to load {filepath}: {e}. Starting fresh.")
            self._data = {}
