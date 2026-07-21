import json
import os
import numpy as np
from typing import Dict, List, Optional, Any

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

class LTMRAGStore:
    """
    Manages the long-term memory (LTM) database for the agent.
    This store holds the textual descriptions of memory signals, their numerical 
    vector embeddings (centroids), and contextual examples (board states). 
    It handles serialization to and from JSON.
    """
    def __init__(self):
        # Maps opponent_name/key -> List of SignalEntry dicts
        # Each dict contains 'name', 'text', 'centroids', and 'examples'
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        
        # Maps opponent_name/key -> raw graveyard string
        # The graveyard stores signals that were deleted (e.g. they failed to work)
        # to ensure the agent doesn't try to reinvent them.
        self.graveyard: Dict[str, str] = {}

    def get_signals(self, key: str) -> List[Dict[str, Any]]:
        """Returns the list of signals for the given key, or [] if none exist."""
        return self.store.get(key, [])

    def get_text(self, key: str, retrieval_log: dict = None) -> str:
        """Returns the flat text blob for the gradient engine, including the graveyard."""
        signals = self.get_signals(key)
        if not signals:
            return "(No signals currently stored)"
            
        text_blocks = []
        for sig in signals:
            block = sig["text"]
            if retrieval_log and sig["name"] in retrieval_log:
                steps = retrieval_log[sig["name"]].get("steps", [])
                if steps:
                    block += f"\n- Retrieved in rounds: {steps}"
            text_blocks.append(block)
            
        main_text = "\n\n".join(text_blocks)
        
        gyard = self.graveyard.get(key, "")
        if gyard.strip():
            return main_text + "\n\n--- GRAVEYARD OF FAILED STRATEGIES ---\n" + gyard.strip()
        return main_text

    def update_signals(self, key: str, signals: List[Dict[str, Any]]) -> None:
        """Replaces the full signal list for the given key."""
        self.store[key] = signals

    def get_graveyard(self, key: str) -> str:
        return self.graveyard.get(key, "")

    def update_graveyard(self, key: str, graveyard_text: str) -> None:
        self.graveyard[key] = graveyard_text

    def add_centroid(self, key: str, signal_name: str, new_vec: np.ndarray, 
                     max_centroids: int = 5, merge_threshold: float = 0.9) -> None:
        """
        Online k-means absorption of a new anchor embedding into centroids.
        Instead of keeping every single board state embedding, we cluster them into up to `max_centroids`.
        This bounds the memory size while maintaining a diverse representation of when a signal applies.
        """
        signals = self.get_signals(key)
        target_sig = next((s for s in signals if s["name"] == signal_name), None)
        if not target_sig:
            return
            
        if "centroids" not in target_sig:
            target_sig["centroids"] = []
            
        centroids = target_sig["centroids"]
        
        # Base case: first centroid
        if not centroids:
            centroids.append({"vec": new_vec.tolist(), "n": 1})
            return
            
        # Find closest centroid
        best_idx = -1
        best_sim = -1.0
        
        for i, c in enumerate(centroids):
            c_vec = np.array(c["vec"])
            sim = cosine_sim(c_vec, new_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                
        if best_sim > merge_threshold:
            # Same region: absorb via running mean
            c = centroids[best_idx]
            old_vec = np.array(c["vec"])
            n = c["n"]
            # new_mean = (old_mean * n + new_vec) / (n + 1)
            updated_vec = (old_vec * n + new_vec) / (n + 1)
            c["vec"] = updated_vec.tolist()
            c["n"] = n + 1
        else:
            # New region
            centroids.append({"vec": new_vec.tolist(), "n": 1})
            if len(centroids) > max_centroids:
                # Merge the two closest centroids to stay within K
                self._merge_closest_centroids(centroids)
                
    def _merge_closest_centroids(self, centroids: List[Dict[str, Any]]) -> None:
        if len(centroids) < 2:
            return
            
        best_pair = (0, 1)
        max_sim = -1.0
        
        for i in range(len(centroids)):
            vec_i = np.array(centroids[i]["vec"])
            for j in range(i + 1, len(centroids)):
                vec_j = np.array(centroids[j]["vec"])
                sim = cosine_sim(vec_i, vec_j)
                if sim > max_sim:
                    max_sim = sim
                    best_pair = (i, j)
                    
        i, j = best_pair
        c1, c2 = centroids[i], centroids[j]
        n1, n2 = c1["n"], c2["n"]
        vec1, vec2 = np.array(c1["vec"]), np.array(c2["vec"])
        
        merged_vec = (vec1 * n1 + vec2 * n2) / (n1 + n2)
        
        # Keep i, delete j
        centroids[i] = {"vec": merged_vec.tolist(), "n": n1 + n2}
        del centroids[j]

    def add_example(self, key: str, signal_name: str, board: str, action: str, 
                    embedder, max_examples: int = 5, redundancy_threshold: float = 0.9) -> None:
        """
        Adds a concrete (board, action) example to the signal's memory.
        Examples help the agent ground the abstract text of the memory into actual game states.
        If we hit the `max_examples` capacity, we evict the most similar existing example (redundancy),
        or the oldest one if they are all diverse.
        """
        signals = self.get_signals(key)
        target_sig = next((s for s in signals if s["name"] == signal_name), None)
        if not target_sig:
            return
            
        if "examples" not in target_sig:
            target_sig["examples"] = []
            
        examples = target_sig["examples"]
        new_example = {"board": board, "action": action}
        
        # If we have space, just append it
        if len(examples) < max_examples:
            examples.append(new_example)
            return
            
        # At capacity: similarity-based eviction
        new_vec = embedder.encode(board)
        
        best_sim = -1.0
        best_idx = -1
        
        for i, ex in enumerate(examples):
            ex_vec = embedder.encode(ex["board"])
            sim = cosine_sim(ex_vec, new_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                
        if best_sim > redundancy_threshold:
            # Redundant pair detected: evict the older (existing) one
            del examples[best_idx]
        else:
            # Diverse: evict the oldest (index 0)
            del examples[0]
            
        # If STILL over capacity (e.g., from unbounded merges in previous code versions), aggressively truncate
        while len(examples) >= max_examples:
            del examples[0]
            
        examples.append(new_example)

    def save(self, filepath: str) -> None:
        """
        Serializes the store and graveyard to a JSON file.
        Uses fsync to ensure the OS flushes the buffer to disk, which is critical 
        for multi-agent environments where different processes might read this file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "signals": self.store,
                "graveyard": self.graveyard
            }, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

    def load(self, filepath: str) -> None:
        """Deserializes the store from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.store = data.get("signals", {})
                    self.graveyard = data.get("graveyard", {})
            except json.JSONDecodeError as e:
                print(f"Warning: JSONDecodeError when loading {filepath}: {e}")
                self.store = {}
                self.graveyard = {}
