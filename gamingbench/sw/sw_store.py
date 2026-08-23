import json
import os
from typing import Dict

class SlidingWindowStore:
    def __init__(self):
        # Maps game_name -> free-form notes string
        self.store: Dict[str, str] = {}

    def get(self, game_name: str):
        """Returns the notes for the given game, or None if no memory exists."""
        return self.store.get(game_name, None)

    def update(self, game_name: str, new_notes: str) -> None:
        """Updates the notes text for the given game."""
        self.store[game_name] = new_notes

    def save(self, filepath: str) -> None:
        """Serializes the SW store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w") as f:
            json.dump({"notes": self.store}, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_filepath, filepath)

    def load(self, filepath: str) -> None:
        """Deserializes the SW store from a JSON file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
            self.store = data.get("notes", {})
