import json
import os
from typing import Dict

class LTMStore:
    def __init__(self):
        # Maps opponent_name -> LTM string
        self.store: Dict[str, str] = {}

    def get(self, opponent_name: str):
        """Returns the LTM for the given opponent, or None if no memory exists."""
        return self.store.get(opponent_name, None)



    def update(self, opponent_name: str, new_ltm: str) -> None:
        """Updates the LTM text for the given opponent."""
        self.store[opponent_name] = new_ltm


    def save(self, filepath: str) -> None:
        """Serializes the LTM store and scores to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump({"ltm": self.store}, f, indent=4)

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
            else:
                self.store = data.get("ltm", {})

# Backward-compatible alias
OpponentLTMStore = LTMStore
