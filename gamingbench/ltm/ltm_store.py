import json
import os
from typing import Dict

class OpponentLTMStore:
    def __init__(self):
        # Maps opponent_name -> LTM string
        self.store: Dict[str, str] = {}
        
    def get(self, opponent_name: str):
        """Returns the LTM for the given opponent, or None if no memory exists."""
        return self.store.get(opponent_name, None)
        
    def update(self, opponent_name: str, new_ltm: str) -> None:
        """Updates the LTM for the given opponent."""
        self.store[opponent_name] = new_ltm
        
    def save(self, filepath: str) -> None:
        """Serializes the LTM store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.store, f, indent=4)
            
    def load(self, filepath: str) -> None:
        """Deserializes the LTM store from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                self.store = json.load(f)
