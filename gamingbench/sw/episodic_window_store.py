import json
import os
from typing import Dict, List
from collections import deque

class EpisodicWindowStore:
    def __init__(self):
        # Maps game_name -> deque of observations
        self.observations: Dict[str, deque] = {}
        # Maps game_name -> synthesized notes string
        self.notes: Dict[str, str] = {}

    def get_notes(self, game_name: str) -> str:
        """Returns the synthesized notes for the given game, or None if no notes exist."""
        return self.notes.get(game_name, None)

    def get_observations(self, game_name: str) -> List[str]:
        """Returns the list of observations for the given game."""
        return list(self.observations.get(game_name, []))

    def add_observation(self, game_name: str, observation: str, window_size: int) -> None:
        """Appends a new observation and ensures the window size is not exceeded."""
        if game_name not in self.observations:
            self.observations[game_name] = deque(maxlen=window_size)
        else:
            # ensure maxlen is set correctly if it wasn't
            if self.observations[game_name].maxlen != window_size:
                self.observations[game_name] = deque(self.observations[game_name], maxlen=window_size)
        self.observations[game_name].append(observation)

    def update_notes(self, game_name: str, new_notes: str) -> None:
        """Updates the synthesized notes text for the given game."""
        self.notes[game_name] = new_notes

    def save(self, filepath: str) -> None:
        """Serializes the EW store to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        serializable_obs = {k: list(v) for k, v in self.observations.items()}
        with open(filepath, "w") as f:
            json.dump({"observations": serializable_obs, "notes": self.notes}, f, indent=4)

    def load(self, filepath: str) -> None:
        """Deserializes the EW store from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            obs_dict = data.get("observations", {})
            self.observations = {k: deque(v) for k, v in obs_dict.items()}
            self.notes = data.get("notes", {})
