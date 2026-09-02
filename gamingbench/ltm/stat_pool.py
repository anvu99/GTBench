import json
import os
import time
import uuid
import threading
import numpy as np
from typing import Dict, Any, List

class StatPool:
    """
    Shared opponent-scoped pool of typed stat trackers.
    Memories hold stat_id references, not stat data.
    
    Schema:
    stats = {
        opponent_key: {
            stat_id: {
                "type": "COUNT" | "RATE" | "DISTRIBUTION" | "MEAN_VAR" | "EXTREMUM",
                "description": "Human-readable semantic description of what is tracked",
                "pseudocode": "Pseudocode defining the tracker logic",
                "vec": [128-dim embedding array],
                "referenced_by": [memory_id1, memory_id2, ...],
                "storage": { ... aggregation data ... }
            }
        }
    }
    
    Aggregation Logic (applied via apply_deltas):
    - COUNT: increments `n` by delta
    - RATE: increments `count` and `total` (e.g. 1 success / 2 attempts)
    - DISTRIBUTION: maintains frequency buckets (e.g. {"aggressive": 5, "passive": 2})
    
    Vector Search:
    When a new stat is proposed, its description is embedded into a vector (`query_vec`).
    `find_relevant_stats` computes cosine similarity between `query_vec` and the 
    `vec` of all existing stats of the same `stat_type` to find candidates for inheritance.
    """
    STAT_TYPES = {"COUNT", "RATE", "MEAN_VAR", "DISTRIBUTION", "EXTREMUM"}

    def __init__(self):
        # Data Schema for the Stat Pool:
        # { 
        #   opponent_key: { 
        #       stat_id: { 
        #           "type": string,                # Enum: COUNT, RATE, MEAN_VAR, DISTRIBUTION, EXTREMUM
        #           "description": string,         # Natural language description of the stat
        #           "pseudocode": string,          # Pseudocode logic of the stat
        #           "storage": dict,               # The running aggregate variables (e.g. n, sum, sum_sq, total, buckets)
        #           "referenced_by": list[string], # Memory IDs that depend on this stat
        #           "vec": list[float]             # The semantic embedding vector of the description
        #       } 
        #   } 
        # }
        self.stats = {}
        self._lock = threading.Lock()

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Manually copy stats and create a fresh lock
        result.stats = copy.deepcopy(self.stats, memo)
        result._lock = threading.Lock()
        
        return result

    def add_stat(self, opp_key: str, stat_type: str, description: str, pseudocode: str, vec: list = None) -> str:
        """Creates a new stat in the pool. Returns the new stat_id."""
        if stat_type not in self.STAT_TYPES:
            raise ValueError(f"Invalid stat type: {stat_type}. Must be one of {self.STAT_TYPES}")

        if opp_key not in self.stats:
            self.stats[opp_key] = {}

        stat_id = f"stat_{uuid.uuid4().hex[:8]}"
        
        storage = {}
        if stat_type == "COUNT":
            storage = {"n": 0}
        elif stat_type == "RATE":
            storage = {"count": 0, "total": 0}
        elif stat_type == "MEAN_VAR":
            storage = {"sum": 0.0, "sum_sq": 0.0, "n": 0}
        elif stat_type == "DISTRIBUTION":
            storage = {"buckets": {}}
        elif stat_type == "EXTREMUM":
            storage = {"max": None, "min": None}

        self.stats[opp_key][stat_id] = {
            "type": stat_type,
            "description": description,
            "pseudocode": pseudocode,
            "storage": storage,
            "referenced_by": [],
            "vec": vec
        }
        
        return stat_id

    def update_stats(self, opp_key: str, mem_id: str, updates: List[Dict]):
        """
        Updates the running values for a set of stats based on the latest game trajectory.
        This handles the mathematical aggregation logic (deltas) for each stat type.
        """
        if opp_key not in self.stats:
            return
        
        for update in updates:
            self.apply_deltas(opp_key, update["stat_id"], update["deltas"])

    def apply_deltas(self, opp_key: str, stat_id: str, deltas: dict):
        """Applies typed delta increments to a stat's storage fields."""
        with self._lock:
            if opp_key not in self.stats or stat_id not in self.stats[opp_key]:
                return

            stat = self.stats[opp_key][stat_id]
            storage = stat["storage"]
            stat_type = stat["type"]

        def _safe_int(val):
            if val is None: return 0
            try: return int(val)
            except (ValueError, TypeError): return 0

        def _safe_float(val):
            if val is None: return 0.0
            try: return float(val)
            except (ValueError, TypeError): return 0.0

        if stat_type == "COUNT":
            storage["n"] += _safe_int(deltas.get("n", 0))
        elif stat_type == "RATE":
            storage["count"] += _safe_int(deltas.get("count", 0))
            storage["total"] += _safe_int(deltas.get("total", 0))
        elif stat_type == "MEAN_VAR":
            storage["sum"] += _safe_float(deltas.get("sum", 0.0))
            storage["sum_sq"] += _safe_float(deltas.get("sum_sq", 0.0))
            storage["n"] += _safe_int(deltas.get("n", 0))
        elif stat_type == "DISTRIBUTION":
            buckets_delta = deltas.get("buckets", {})
            for k, v in buckets_delta.items():
                k_str = str(k)
                try:
                    storage["buckets"][k_str] = storage["buckets"].get(k_str, 0) + int(v)
                except (TypeError, ValueError):
                    pass
        elif stat_type == "EXTREMUM":
            d_max = deltas.get("max")
            d_min = deltas.get("min")
            if d_max is not None:
                if storage["max"] is None or d_max > storage["max"]:
                    storage["max"] = d_max
            if d_min is not None:
                if storage["min"] is None or d_min < storage["min"]:
                    storage["min"] = d_min

    def add_reference(self, opp_key: str, stat_id: str, memory_id: str):
        """Records that a memory references this stat."""
        if opp_key in self.stats and stat_id in self.stats[opp_key]:
            if memory_id not in self.stats[opp_key][stat_id]["referenced_by"]:
                self.stats[opp_key][stat_id]["referenced_by"].append(memory_id)

    def remove_reference(self, opp_key: str, stat_id: str, memory_id: str):
        """Removes reference. Garbage-collects stat if referenced_by becomes empty."""
        if opp_key in self.stats and stat_id in self.stats[opp_key]:
            refs = self.stats[opp_key][stat_id]["referenced_by"]
            if memory_id in refs:
                refs.remove(memory_id)
            if len(refs) == 0:
                del self.stats[opp_key][stat_id] # GC

    def format_for_injection(self, opp_key: str, stat_ids: list) -> str:
        """Formats stats as human-readable block with computed values (%, mean, std, mode)."""
        if opp_key not in self.stats or not stat_ids:
            return ""

        formatted_stats = []
        for sid in stat_ids:
            if sid not in self.stats[opp_key]:
                continue
            
            stat = self.stats[opp_key][sid]
            stype = stat["type"]
            sdesc = stat.get("description", "")
            spseudo = stat.get("pseudocode", "")
            st = stat["storage"]
            
            val_str = ""
            if stype == "COUNT":
                val_str = f"n={st['n']}"
            elif stype == "RATE":
                cnt, tot = st["count"], st["total"]
                if tot == 0:
                    val_str = "0% (0/0 rounds)"
                else:
                    pct = (cnt / tot) * 100
                    val_str = f"{pct:.1f}% ({cnt}/{tot} rounds)"
            elif stype == "MEAN_VAR":
                s, s_sq, n = st["sum"], st["sum_sq"], st["n"]
                if n == 0:
                    val_str = "mean=N/A, std=N/A (n=0)"
                else:
                    mean = s / n
                    var = max(0, (s_sq / n) - (mean ** 2))
                    std = np.sqrt(var)
                    val_str = f"mean={mean:.2f}, std={std:.2f} (n={n})"
            elif stype == "DISTRIBUTION":
                b = st["buckets"]
                if not b:
                    val_str = "No data"
                else:
                    mode = max(b, key=b.get)
                    total_n = sum(b.values())
                    
                    try:
                        sum_val = sum(float(k) * v for k, v in b.items())
                        mean = sum_val / total_n
                        val_str = f"{b} — mode={mode}, mean={mean:.2f}"
                    except ValueError:
                        # Buckets aren't numbers
                        val_str = f"{b} — mode={mode}"
            elif stype == "EXTREMUM":
                val_str = f"max={st['max']}, min={st['min']}"

            # Use original description, but provide exact values
            formatted_stats.append(f"[{sid}] {stype}: {sdesc}\n    Current Values: {val_str}")
            
        if not formatted_stats:
            return ""
            
        return "\n".join(formatted_stats)


    def find_relevant_stats(self, opp_key: str, query_vec: list, top_k: int = 3, stat_type: str = None) -> List[str]:
        """
        Performs a semantic vector search across existing stats.
        This is the core of the scaling mechanism, preventing redundancy by proposing
        existing stats that semantically match a newly proposed tracker description.
        """
        if opp_key not in self.stats or not self.stats[opp_key]:
            return []
            
        candidates = []
        for sid, stat in self.stats[opp_key].items():
            if stat_type and stat.get("type") != stat_type:
                continue
            
            vec = stat.get("vec")
            if vec is not None:
                vec_np = np.array(vec)
                norm_query = np.linalg.norm(query_vec)
                norm_vec = np.linalg.norm(vec_np)
                if norm_query == 0 or norm_vec == 0:
                    sim = 0
                else:
                    sim = np.dot(query_vec, vec_np) / (norm_query * norm_vec)
                candidates.append((sim, sid))
                
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [sid for sim, sid in candidates[:top_k]]
    def format_pool_summary(self, opp_key: str, candidate_ids: list = None) -> str:
        """Compact list for inheritance LLM call. E.g.: 'stat_abc | RATE | Cap breach rate | n=47'"""
        if opp_key not in self.stats:
            return "No existing stats for this opponent."

        summary_lines = []
        for sid, stat in self.stats[opp_key].items():
            if candidate_ids is not None and sid not in candidate_ids:
                continue
            stype = stat["type"]
            sdesc = stat.get("description", "")
            spseudo = stat.get("pseudocode", "")
            st = stat["storage"]
            n_obs = 0
            if stype == "COUNT": n_obs = st["n"]
            elif stype == "RATE": n_obs = st["total"]
            elif stype == "MEAN_VAR": n_obs = st["n"]
            elif stype == "DISTRIBUTION": n_obs = sum(st["buckets"].values()) if "buckets" in st else 0
            elif stype == "EXTREMUM": n_obs = "N/A"
            
            summary_lines.append(f"{sid} | {stype} | {sdesc} | n_obs={n_obs}")

        return "\n".join(summary_lines)

    def save(self, filepath: str) -> None:
        """Serializes the stat pool to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"stats": self.stats}, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

    def load(self, filepath: str) -> None:
        """Deserializes the stat pool from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stats = data.get("stats", {})
            except json.JSONDecodeError as e:
                print(f"Warning: JSONDecodeError when loading {filepath}: {e}")
                self.stats = {}
