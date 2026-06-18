import json
import os
from typing import Dict

class LTMStore:
    def __init__(self):
        # Maps opponent_name -> LTM string
        self.store: Dict[str, str] = {}
        # Maps opponent_name -> {signal_name -> confidence_score (float in [0,1])}
        self.scores: Dict[str, Dict[str, float]] = {}

    def get(self, opponent_name: str):
        """Returns the LTM for the given opponent, or None if no memory exists."""
        return self.store.get(opponent_name, None)

    def get_scores(self, opponent_name: str) -> Dict[str, float]:
        """Returns the confidence score dict for the given opponent, or {} if none."""
        return dict(self.scores.get(opponent_name, {}))

    def update(self, opponent_name: str, new_ltm: str) -> None:
        """Updates the LTM text for the given opponent."""
        self.store[opponent_name] = new_ltm

    def update_scores(self, opponent_name: str, scores_dict: Dict[str, float]) -> None:
        """Overwrites the confidence scores for the given opponent."""
        self.scores[opponent_name] = dict(scores_dict)

    def save(self, filepath: str) -> None:
        """Serializes the LTM store and scores to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"ltm": self.store, "scores": self.scores}, f, indent=4)

    def load(self, filepath: str) -> None:
        """Deserializes the LTM store from a JSON file.
        
        Supports both the new format {"ltm": {...}, "scores": {...}}
        and the legacy format {opponent_name: ltm_text}.
        """
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            # Detect legacy format: values are strings, not dicts
            if data and all(isinstance(v, str) for v in data.values()):
                self.store = data
                self.scores = {}
            else:
                self.store = data.get("ltm", {})
                self.scores = data.get("scores", {})

# Backward-compatible alias
OpponentLTMStore = LTMStore
