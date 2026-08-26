import json
import os
from typing import Dict, Tuple, Optional

class AgentProStore:
    def __init__(self):
        # Maps game_name -> opponent_key -> {"belief": dict, "summary": dict}
        self.data: Dict[str, Dict[str, Dict]] = {}

    def get(self, game_name: str, opponent_key: str) -> Tuple[Dict, Dict]:
        """Returns the belief and summary for the given game and opponent.
        If none exist, returns empty defaults matching Agent-Pro."""
        game_data = self.data.get(game_name, {})
        opp_data = game_data.get(opponent_key, {})
        
        belief = opp_data.get("belief", {"ses": "", "ops": "", "opo": ""})
        summary = opp_data.get("summary", {"rea": "", "ref": ""})
        
        return belief, summary

    def update(self, game_name: str, opponent_key: str, belief: Dict, summary: Dict) -> None:
        """Updates the belief and summary for the given game and opponent."""
        if game_name not in self.data:
            self.data[game_name] = {}
            
        self.data[game_name][opponent_key] = {
            "belief": belief,
            "summary": summary
        }

    def save(self, filepath: str) -> None:
        """Serializes the AgentPro store to a JSON file atomically."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = filepath + ".tmp"
        with open(tmp_filepath, "w") as f:
            json.dump(self.data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_filepath, filepath)

    def load(self, filepath: str) -> None:
        """Deserializes the AgentPro store from a JSON file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self.data = json.load(f)
            except json.JSONDecodeError:
                self.data = {}
