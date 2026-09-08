"""
gamingbench/ltm/cpqa_obs_graph.py

Observation Graph Store for CPQA v2.

Replaces cpqa_store.py. Per-opponent schema:
  - observation_nodes: typed behavioral observations with counts and component membership
  - faiss_index_data: per-type anchor-field embedding indexes (for similarity retrieval)
  - memories: synthesized memory entries (connected components), each with bullet points + counts
  - consolidated_block: rendered profile injected into every game prompt
  - games_observed: total games seen against this opponent

Three observation types:
  UNCONDITIONAL  — general behavior regardless of game state
  CONDITIONAL    — behavior given a specific game state / trigger
  INFERENCE      — inferred private information / underlying driver from a public action
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


def _make_obs_id() -> str:
    return f"obs_{uuid.uuid4().hex[:8]}"


def _make_mem_id() -> str:
    return f"mem_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Observation node helpers
# ---------------------------------------------------------------------------

def _obs_anchor_text(node: dict) -> str:
    """Returns the anchor field text used for FAISS indexing (by type)."""
    t = node.get("type", "")
    if t == "UNCONDITIONAL":
        return node.get("behavior", "")
    elif t == "CONDITIONAL":
        return node.get("game_state", "")
    elif t == "INFERENCE":
        return node.get("public_action", "")
    return ""


def _obs_full_text(node: dict) -> str:
    """Returns a full-text representation of the observation for dedup."""
    t = node.get("type", "")
    if t == "UNCONDITIONAL":
        return node.get("behavior", "")
    elif t == "CONDITIONAL":
        gs = node.get("game_state", "")
        act = node.get("action", "")
        return f"{gs} | {act}"
    elif t == "INFERENCE":
        pa = node.get("public_action", "")
        driver = node.get("underlying_driver", "")
        return f"INFERENCE: Action: {pa}, Driver: {driver}"
    return ""


def _format_memory_for_prompt(mem: dict) -> str:
    """Formats a memory entry for injection into batch prompts."""
    lines = [
        f"Memory ID: {mem['id']} | Type: {mem['type']}",
        f"Name: {mem.get('name', '')}",
        f"Description: {mem.get('description', '')}",
    ]
    anchor = mem.get("anchor", "")
    if anchor:
        label = "Game State" if mem["type"] == "CONDITIONAL" else "Public Action"
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
    Observation graph store for CPQA v2.

    Thread-safe. Each opponent has:
      - observation_nodes: dict[obs_id -> node]
      - faiss_index_data: per-type lists of (obs_id, embedding) for rebuild
      - memories: dict[mem_id -> memory_entry]
      - consolidated_block: str
      - games_observed: int
    """

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        # Live FAISS indexes (rebuilt on load, not persisted directly)
        self._faiss_indexes: Dict[str, Dict[str, Any]] = {}  # opp_key -> {type -> index}

    # ------------------------------------------------------------------
    # Opponent scaffolding
    # ------------------------------------------------------------------

    def _ensure_opponent(self, opp_key: str):
        if opp_key not in self._data:
            self._data[opp_key] = {
                "observation_nodes": {},
                "faiss_index_data": {
                    "UNCONDITIONAL": [],
                    "CONDITIONAL": [],
                    "INFERENCE": [],
                },
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
    # Observation node CRUD
    # ------------------------------------------------------------------

    def add_observation_node(self, opp_key: str, obs_type: str, fields: dict,
                             embedding: Optional[np.ndarray] = None,
                             count: int = 1) -> str:
        """
        Inserts a new observation node. Returns obs_id.

        fields for UNCONDITIONAL: {"behavior": str}
        fields for CONDITIONAL:   {"game_state": str, "action": str}
        fields for INFERENCE:     {"public_action": str, "underlying_driver": str}
        """
        if obs_type not in OBS_TYPES:
            raise ValueError(f"Invalid obs_type: {obs_type}")
        self._ensure_opponent(opp_key)
        obs_id = _make_obs_id()
        node = {
            "id": obs_id,
            "type": obs_type,
            "count": count,
            "component_id": None,   # set after edge creation / synthesis
            "created_at": time.time(),
            "updated_at": time.time(),
            **fields,
        }
        emb_list = embedding.tolist() if embedding is not None else None
        node["embedding"] = emb_list

        with self._lock:
            self._data[opp_key]["observation_nodes"][obs_id] = node
            if emb_list is not None:
                self._data[opp_key]["faiss_index_data"][obs_type].append({
                    "obs_id": obs_id,
                    "vec": emb_list,
                })
                self._rebuild_faiss_index(opp_key, obs_type)

        return obs_id

    def get_observation_node(self, opp_key: str, obs_id: str) -> Optional[dict]:
        return self._data.get(opp_key, {}).get("observation_nodes", {}).get(obs_id)

    def get_all_observation_nodes(self, opp_key: str, obs_type: str = None) -> List[dict]:
        nodes = list(self._data.get(opp_key, {}).get("observation_nodes", {}).values())
        if obs_type:
            nodes = [n for n in nodes if n.get("type") == obs_type]
        return nodes

    def increment_observation_count(self, opp_key: str, obs_id: str, delta: int = 1):
        node = self.get_observation_node(opp_key, obs_id)
        if node:
            with self._lock:
                node["count"] = node.get("count", 1) + delta
                node["updated_at"] = time.time()

    def set_observation_component(self, opp_key: str, obs_id: str, mem_id: str):
        node = self.get_observation_node(opp_key, obs_id)
        if node:
            node["component_id"] = mem_id

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def _rebuild_faiss_index(self, opp_key: str, obs_type: str):
        """Rebuilds the FAISS flat index for a given type from stored vectors."""
        if not FAISS_AVAILABLE:
            return
        entries = self._data[opp_key]["faiss_index_data"].get(obs_type, [])
        if not entries:
            self._faiss_indexes[opp_key][obs_type] = None
            return
        dim = len(entries[0]["vec"])
        index = faiss.IndexFlatIP(dim)
        vecs = np.array([e["vec"] for e in entries], dtype=np.float32)
        # L2-normalize for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = vecs / norms
        index.add(vecs)
        self._faiss_indexes[opp_key][obs_type] = {
            "index": index,
            "obs_ids": [e["obs_id"] for e in entries],
        }

    def query_similar_nodes(self, opp_key: str, obs_type: str,
                             query_vec: np.ndarray, top_k: int = 5,
                             threshold: float = 0.0) -> List[dict]:
        """
        Returns up to top_k similar existing observation nodes for a given type,
        sorted by cosine similarity descending. Only returns nodes with
        similarity >= threshold.

        Returns list of {"obs_id": str, "score": float, "node": dict}
        """
        self._ensure_opponent(opp_key)
        idx_data = self._faiss_indexes.get(opp_key, {}).get(obs_type)
        if idx_data is None or not FAISS_AVAILABLE:
            return []

        index = idx_data["index"]
        obs_ids = idx_data["obs_ids"]

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
            if idx < 0:
                continue
            if score < threshold:
                continue
            obs_id = obs_ids[idx]
            node = self.get_observation_node(opp_key, obs_id)
            if node:
                results.append({"obs_id": obs_id, "score": float(score), "node": node})
        return results

    def get_component_memory_for_node(self, opp_key: str, obs_id: str) -> Optional[dict]:
        """Returns the memory entry for the component that obs_id belongs to."""
        node = self.get_observation_node(opp_key, obs_id)
        if not node or not node.get("component_id"):
            return None
        return self.get_memory(opp_key, node["component_id"])

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
                      observation_ids: List[str] = None) -> str:
        """
        Creates or fully replaces a memory entry. Returns mem_id.

        bullets: list of {"id": str, "description": str, "count": int}
        observation_ids: obs_ids that belong to this component
        """
        self._ensure_opponent(opp_key)
        if mem_id is None:
            mem_id = _make_mem_id()
        with self._lock:
            self._data[opp_key]["memories"][mem_id] = {
                "id": mem_id,
                "type": mem_type,
                "name": name,
                "description": description,
                "anchor": anchor,
                "bullets": bullets,
                "observation_ids": observation_ids or [],
                "updated_at": time.time(),
            }
        return mem_id

    def increment_bullet_count(self, opp_key: str, mem_id: str, bullet_id: str, delta: int = 1):
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

    def add_bullet_to_memory(self, opp_key: str, mem_id: str, description: str) -> Optional[str]:
        """
        Appends a new bullet to a memory entry. Auto-assigns next letter ID.
        Returns the new bullet ID, or None if memory not found.
        """
        mem = self.get_memory(opp_key, mem_id)
        if not mem:
            return None
        with self._lock:
            bullets = mem.get("bullets", [])
            next_id = chr(ord('A') + len(bullets))
            bullets.append({"id": next_id, "description": description, "count": 1})
            mem["bullets"] = bullets
            mem["updated_at"] = time.time()
        return next_id

    def assign_obs_to_component(self, opp_key: str, obs_ids: List[str], mem_id: str):
        """Sets component_id on a list of observation nodes."""
        for obs_id in obs_ids:
            self.set_observation_component(opp_key, obs_id, mem_id)
        # Also update the memory's observation_ids list
        mem = self.get_memory(opp_key, mem_id)
        if mem:
            with self._lock:
                existing = set(mem.get("observation_ids", []))
                existing.update(obs_ids)
                mem["observation_ids"] = list(existing)

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

    def format_candidate_memories_for_obs(self, opp_key: str, obs_ids: List[str]) -> str:
        """
        Given a list of observation node IDs (FAISS neighbors),
        resolves their component memories (deduplicated) and formats them
        for the batch dedup+edge prompt.
        """
        seen_mem_ids = set()
        blocks = []
        for obs_id in obs_ids:
            mem = self.get_component_memory_for_node(opp_key, obs_id)
            if mem and mem["id"] not in seen_mem_ids:
                seen_mem_ids.add(mem["id"])
                blocks.append(_format_memory_for_prompt(mem))
        if not blocks:
            return "(No related existing components found — this may start a new component.)"
        return "\n\n---\n\n".join(blocks)

    def format_component_for_synthesis(self, opp_key: str, mem_id: Optional[str],
                                        obs_ids: List[str]) -> str:
        """
        Formats the observation nodes of a component for the memory synthesis prompt.
        """
        lines = []
        for obs_id in obs_ids:
            node = self.get_observation_node(opp_key, obs_id)
            if not node:
                continue
            t = node["type"]
            if t == "UNCONDITIONAL":
                lines.append(f"  - [count: {node['count']}] {node.get('behavior', '')}")
            elif t == "CONDITIONAL":
                lines.append(
                    f"  - [count: {node['count']}] Trigger: {node.get('game_state', '')} | "
                    f"Action: {node.get('action', '')}"
                )
            elif t == "INFERENCE":
                lines.append(
                    f"  - [count: {node['count']}] Public Action: {node.get('public_action', '')} | "
                    f"Underlying Driver: {node.get('underlying_driver', '')}"
                )
        return "\n".join(lines) if lines else "(empty component)"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Store embedding data inside the main JSON (as lists, not numpy)
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
            # Rebuild FAISS indexes from stored vectors
            for opp_key in self._data:
                self._ensure_opponent(opp_key)
                for obs_type in OBS_TYPES:
                    entries = self._data[opp_key].get("faiss_index_data", {}).get(obs_type, [])
                    if entries:
                        self._rebuild_faiss_index(opp_key, obs_type)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Warning: ObsGraphStore failed to load {filepath}: {e}. Starting fresh.")
            self._data = {}
