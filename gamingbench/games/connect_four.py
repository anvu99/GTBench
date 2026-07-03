
import copy
from gamingbench.games.openspiel_adapter import OpenSpielGame
import re


class ConnectFour(OpenSpielGame):
    def __init__(self) -> None:
        super().__init__("connect_four")
        self.game_name = 'connect4'
        pass

    def openspiel_action_to_agent(self, action):

        actions = [f'<C{int(a[1]) + 1}>' for a in action]
        return actions
        pass

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        opponent_idx = 1 if current_player_idx == 0 else 0
        lines = openspiel_obs.strip().split('\n')
        formatted_rows = []
        for r_idx, line in enumerate(lines):
            row_num = 6 - r_idx
            row_parts = []
            for c_idx, char in enumerate(line.strip()):
                col_num = c_idx + 1
                row_parts.append(f"C{col_num}R{row_num}={char}")
            row_label = f"Row {row_num}"
            if row_num == 6:
                row_label += " (top)"
            elif row_num == 1:
                row_label += " (bottom)"
            formatted_rows.append(f"{row_label}: " + ", ".join(row_parts))
        board_preview = "\n".join(formatted_rows)
        res = {
            'board': board_preview,
            'opponent_moves': copy.deepcopy(self.quick_action_memory_for_llm.get(opponent_idx, [])),
            'self_moves': copy.deepcopy(self.quick_action_memory_for_llm.get(current_player_idx, [])),
            'player_idx': current_player_idx,
        }
        return res
        pass

    def agent_action_to_openspiel(self, action):
        try:
            regex = r"\s*(\d+)\s*"
            numbers_match = re.findall(
                regex, action)
            column = int(numbers_match[0])-1
            return min(column, 6)
        except Exception as e:

            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None
        pass
