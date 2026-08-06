import json
import os
import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Any

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a) # Calculate magnitude of vector a
    b_norm = np.linalg.norm(b) # Calculate magnitude of vector b
    if a_norm == 0 or b_norm == 0:
        return 0.0 # Prevent division by zero if either vector is completely zero
    return float(np.dot(a, b) / (a_norm * b_norm)) # Return the dot product normalized by the magnitudes

class TwoLayerStore:
    """
    Manages the 2-layer memory architecture for EvidenceMemoryAgent and ProactiveQueryAgent.
    Layer 1: In-game Evidence (raw objective observations extracted from the game)
    Layer 2: Consolidated Memories (high-level strategic insights, embedded for vector retrieval)
    """
    def __init__(self):
        # Layer 1: Raw semantic evidence extracted from game trajectories.
        # Structure: { opponent_key: { evidence_id: {content, observation, game_id, vec} } }
        self.evidence: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Layer 2: Synthesized strategic memories.
        # Structure: { opponent_key: { memory_id: {content, evidence_ids_used, score, vec} } }
        self.memories: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_evidence(self, key: str, content: str, observation: str, game_id: str, vec: Optional[List[float]] = None, max_total_evidence: int = 500) -> str:
        """Adds a new evidence item to Layer 1 and returns its newly generated ID."""
        # Initialize the nested dict for this specific opponent if it doesn't exist yet
        if key not in self.evidence:
            self.evidence[key] = {}
        
        # Generate a unique 8-character hex ID prefixed with 'ev_' for this evidence
        evidence_id = f"ev_{uuid.uuid4().hex[:8]}"
        
        # Store the raw evidence block containing its content, where it came from, and timestamps
        self.evidence[key][evidence_id] = {
            "id": evidence_id,
            "content": content,
            "observation": observation,
            "game_id": game_id,
            "timestamp": time.time(),
            "vec": vec
        }
        
        # Enforce maximum evidence limit (Layer 1 capacity)
        if len(self.evidence[key]) > max_total_evidence:
            # Sort evidence IDs by their timestamp (oldest first)
            sorted_ev = sorted(self.evidence[key].values(), key=lambda x: x["timestamp"])
            # Calculate how many items to trim
            excess_count = len(self.evidence[key]) - max_total_evidence
            # Remove the oldest excess items from the dictionary
            for ev_to_remove in sorted_ev[:excess_count]:
                del self.evidence[key][ev_to_remove["id"]]
                
        return evidence_id

    def get_evidence(self, key: str, evidence_id: str) -> Optional[Dict[str, Any]]:
        """Safely retrieves a specific evidence item for an opponent, returning None if missing."""
        return self.evidence.get(key, {}).get(evidence_id)

    def get_all_evidence(self, key: str) -> List[Dict[str, Any]]:
        """Returns a list of all Layer 1 evidence items for an opponent."""
        return list(self.evidence.get(key, {}).values())

    def find_relevant_evidence(self, key: str, query_vec: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant evidence items using cosine similarity against the query vector."""
        evidence_list = self.get_all_evidence(key)
        if not evidence_list:
            return []
            
        scored_evidence = []
        for ev in evidence_list:
            vec_list = ev.get("vec")
            if vec_list is not None:
                ev_vec = np.array(vec_list)
                sim = cosine_sim(query_vec, ev_vec)
                scored_evidence.append((sim, ev))
                
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        return [ev for sim, ev in scored_evidence[:top_k]]

    def get_memories(self, key: str) -> List[Dict[str, Any]]:
        """Returns the list of all Layer 2 consolidated memories for an opponent."""
        return self.memories.get(key, [])

    def get_memory(self, key: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Iterates through an opponent's memories to find and return a specific memory by its ID."""
        for mem in self.memories.get(key, []):
            if mem["id"] == memory_id:
                return mem
        return None

    def add_memory(self, key: str, content: str, evidence_ids: List[str], vec: np.ndarray, max_evidence_per_memory: int = 6, question: str = None) -> str:
        """Creates a new Level 2 memory, seeded by the provided list of evidence IDs."""
        # Initialize the memory list for this opponent if this is their first memory
        if key not in self.memories:
            self.memories[key] = []
            
        # Generate a unique 8-character hex ID prefixed with 'mem_' for the new memory
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        
        # Enforce FIFO (First-In, First-Out) on the evidence list so it doesn't grow indefinitely
        # We only keep the most recent `max_evidence_per_memory` IDs
        evidence_ids = evidence_ids[-max_evidence_per_memory:]
        
        # Append the new memory to the list with its embedded vector (for future cosine similarity retrieval)
        self.memories[key].append({
            "id": memory_id,
            "content": content,
            "evidence_ids": evidence_ids,
            "created_at": time.time(),
            "updated_at": time.time(),
            "generation": 1,
            "question": question,
            "score": 0,
            "vec": vec.tolist() if vec is not None else None # Store the numpy array as a JSON-serializable list
        })
        return memory_id

    def update_memory(self, key: str, memory_id: str, new_content: str, new_evidence_ids: List[str], vec: np.ndarray, max_evidence_per_memory: int = 6, question: str = None) -> bool:
        """Updates an existing Level 2 memory, appending new evidence while enforcing the FIFO limit."""
        # Find the memory to update
        mem = self.get_memory(key, memory_id)
        if not mem:
            return False # Return False if the memory doesn't exist
            
        # Overwrite the old content with the new, LLM-generated updated content
        mem["content"] = new_content
        
        # Optionally overwrite the question if one is provided
        if question is not None:
            mem["question"] = question
        
        # Combine the old evidence IDs with the newly appended ones
        combined_ev = mem["evidence_ids"] + new_evidence_ids
        
        # Deduplicate the evidence IDs while preserving their most recent chronological order.
        # If an old evidence ID is re-added, this moves it to the back (newest position)
        # so it doesn't get evicted by the FIFO cap.
        dedup_ev = []
        for eid in combined_ev:
            if eid in dedup_ev:
                dedup_ev.remove(eid)
            dedup_ev.append(eid)
                
        # Enforce the FIFO cap, discarding the oldest evidence IDs if we exceed the limit
        mem["evidence_ids"] = dedup_ev[-max_evidence_per_memory:]
        
        # Update metadata to track changes
        mem["updated_at"] = time.time()
        mem["generation"] += 1
        
        # Update the embedded vector so future retrievals match the new content
        if vec is not None:
            mem["vec"] = vec.tolist()
            
        return True

    def find_relevant_memories(self, key: str, query_vec: np.ndarray, top_k: int = 6) -> List[Dict[str, Any]]:
        """Retrieves top-k relevant memories using cosine similarity against the query vector."""
        # Grab all memories for the opponent
        memories = self.get_memories(key)
        if not memories:
            return []
            
        scored_memories = []
        # Score each memory by comparing the query's vector to the memory's stored vector
        for mem in memories:
            if "vec" in mem and mem["vec"] is not None:
                mem_vec = np.array(mem["vec"]) # Convert the stored JSON list back into a numpy array
                sim = cosine_sim(mem_vec, query_vec) # Calculate similarity score
                scored_memories.append((sim, mem))
                
        # Sort in descending order (highest similarity first)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        # Strip the scores and return just the top-k memory dictionaries
        return [mem for sim, mem in scored_memories[:top_k]]

    def update_score(self, key: str, memory_id: str, delta: int) -> bool:
        """Updates the reliability score of a specific memory."""
        mem = self.get_memory(key, memory_id)
        if not mem:
            return False
        
        # Ensure the field exists for older memories loaded from disk
        if "score" not in mem:
            mem["score"] = 0
            
        mem["score"] += delta
        return True

    def get_top_k_by_score(self, key: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Returns the top-k memories sorted by their reliability score (descending)."""
        memories = self.get_memories(key)
        if not memories:
            return []
            
        # Filter out memories that don't have a question (legacy or EMA memories)
        valid_mems = [m for m in memories if m.get("question")]
        valid_mems.sort(key=lambda m: m.get("score", 0), reverse=True)
        return valid_mems[:top_k]

    def save(self, filepath: str) -> None:
        """Serializes the dual-layer store to a JSON file."""
        # Ensure the target directory exists before writing
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            # Write both the evidence dictionary and the memories list into one combined file
            json.dump({
                "evidence": self.evidence,
                "memories": self.memories
            }, f, indent=4) # Use indent=4 for human-readable JSON output
            # Force OS to write buffers to disk to prevent data loss on crash
            f.flush()
            os.fsync(f.fileno())

    def load(self, filepath: str) -> None:
        """Deserializes the store from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Safely extract the layers, defaulting to empty dicts if not found
                    self.evidence = data.get("evidence", {})
                    self.memories = data.get("memories", {})
            except json.JSONDecodeError as e:
                # If the JSON is corrupted, start fresh rather than crashing
                print(f"Warning: JSONDecodeError when loading {filepath}: {e}")
                self.evidence = {}
                self.memories = {}
