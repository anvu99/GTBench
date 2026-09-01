import json
import os
import time
import uuid
from typing import Dict, List, Optional, Any, Literal

class StrategyStore:
    """
    Manages a global strategy memory bank for ProactiveQueryAgent.
    Stores agent-generated strategies with self-defined success/neutral/failure criteria.
    Strategies are ranked by performance (success_count - failure_count) and injected into prompts.
    """
    def __init__(self):
        # Structure: { strategy_id: { id, title, definition, success_criteria, neutral_criteria, failure_criteria, success_count, neutral_count, failure_count, total_score, created_at } }
        self.strategies: Dict[str, Dict[str, Any]] = {}

    def add_strategy(self, title: str, strategic_reasoning: str, tactical_guidance: str, desired_post_game_reflection: str) -> str:
        """Adds a new strategy and returns its generated ID."""
        strategy_id = f"strat_{uuid.uuid4().hex[:8]}"
        
        self.strategies[strategy_id] = {
            "id": strategy_id,
            "title": title,
            "strategic_reasoning": strategic_reasoning,
            "tactical_guidance": tactical_guidance,
            "desired_post_game_reflection": desired_post_game_reflection,
            "recent_reflections": [],
            "recent_execution_log": "No execution data yet. This strategy has not been tested.",
            "success_count": 0,
            # "neutral_count": 0,
            "failure_count": 0,
            "total_score": 0,
            "uses_count": 0,
            "total_utility": 0.0,
            "average_utility": 0.0,
            "created_at": time.time()
        }
        return strategy_id

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_all(self) -> List[Dict[str, Any]]:
        """Returns a list of all stored strategies."""
        return list(self.strategies.values())

    def update_score(self, strategy_id: str, outcome: Literal["success", "failure"]) -> bool:
        """Increments the appropriate counter for a strategy and updates its total score."""
        strat = self.get_strategy(strategy_id)
        if not strat:
            return False
            
        if outcome == "success":
            strat["success_count"] += 1
        # elif outcome == "neutral":
        #     strat["neutral_count"] += 1
        elif outcome == "failure":
            strat["failure_count"] += 1
        else:
            return False
            
        # Total score = successes - failures (neutrals don't affect rank directly)
        strat["total_score"] = strat["success_count"] - strat["failure_count"]
        return True

    def update_utility(self, strategy_id: str, utility: float) -> bool:
        """Updates the empirical utility and calculates the average utility."""
        strat = self.get_strategy(strategy_id)
        if not strat:
            return False
            
        strat["uses_count"] = strat.get("uses_count", 0) + 1
        strat["total_utility"] = strat.get("total_utility", 0.0) + utility
        strat["average_utility"] = strat["total_utility"] / strat["uses_count"]
        return True

    def add_reflection(self, strategy_id: str, reflection: str, max_queue_size: int = 5) -> bool:
        """Pushes a new reflection observation to the FIFO queue."""
        strat = self.get_strategy(strategy_id)
        if not strat:
            return False
            
        if "recent_reflections" not in strat:
            strat["recent_reflections"] = []
            
        strat["recent_reflections"].insert(0, reflection)
        if len(strat["recent_reflections"]) > max_queue_size:
            strat["recent_reflections"].pop()
        return True
        
    def update_execution_log(self, strategy_id: str, summary: str) -> bool:
        """Updates the recent_execution_log string."""
        strat = self.get_strategy(strategy_id)
        if not strat:
            return False
            
        strat["recent_execution_log"] = summary
        return True

    def get_top_k_by_score(self, top_k: int = 6) -> List[Dict[str, Any]]:
        """Returns the top-k strategies sorted by their average_utility (descending)."""
        strats = self.get_all()
        # Sort by average_utility descending, then by created_at descending (newest first for ties)
        strats.sort(key=lambda s: (s.get("average_utility", 0.0), s.get("created_at", 0)), reverse=True)
        return strats[:top_k]
        
    def get_bottom_k_by_utility(self, k: int = 3) -> List[Dict[str, Any]]:
        """Returns the bottom-k strategies sorted by average_utility (ascending) to use as anti-patterns."""
        strats = [s for s in self.get_all() if s.get("uses_count", 0) > 0]
        # Sort ascending by average utility
        strats.sort(key=lambda s: (s.get("average_utility", 0.0), -s.get("created_at", 0)))
        return strats[:k]

    def get_mixed_top_k(self, top_score_k: int = 4, top_recent_k: int = 2) -> List[Dict[str, Any]]:
        """Returns top_score_k strategies by score, plus top_recent_k strategies by recency."""
        strats = self.get_all()
        # Sort by average_utility descending
        strats.sort(key=lambda s: (s.get("average_utility", 0.0), s.get("created_at", 0)), reverse=True)
        top_by_score = strats[:top_score_k]
        
        # Remaining strategies
        picked_ids = {s['id'] for s in top_by_score}
        remaining = [s for s in strats if s['id'] not in picked_ids]
        
        # Sort remaining by created_at descending (newest first)
        remaining.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        top_by_recent = remaining[:top_recent_k]
        
        return top_by_score + top_by_recent

    def save(self, filepath: str) -> None:
        """Serializes the strategy store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "strategies": self.strategies
            }, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

    def load(self, filepath: str) -> None:
        """Deserializes the strategy store from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.strategies = data.get("strategies", {})
            except json.JSONDecodeError as e:
                print(f"Warning: JSONDecodeError when loading {filepath}: {e}")
                self.strategies = {}
