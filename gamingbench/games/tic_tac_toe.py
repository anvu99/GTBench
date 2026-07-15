import numpy as np
from typing import List
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils
import copy
from gamingbench.games.openspiel_adapter import OpenSpielGame
import re


class TicTacToe(OpenSpielGame):
    def __init__(self, config=None) -> None:
        super().__init__("tic_tac_toe")
        self.game_name = 'tictactoe'
        pass

    def openspiel_action_to_agent(self, action):

        actions = [f'<C{int(a[4])+1}R{int(a[2])+1}>' for a in action]
        return actions
        pass

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        opponent_idx = 1 if current_player_idx == 0 else 0
        lines = openspiel_obs.strip().split('\n')
        formatted_rows = []
        for r_idx, line in enumerate(lines):
            row_num = r_idx + 1
            row_parts = []
            for c_idx, char in enumerate(line.strip()):
                col_num = c_idx + 1
                row_parts.append(f"C{col_num}R{row_num}={char}")
            formatted_rows.append(f"Row {row_num}: " + ", ".join(row_parts))
        board_preview = "\n".join(formatted_rows)
        res = {
            'board': board_preview,
            'opponent_moves': copy.deepcopy(self.quick_action_memory_for_llm.get(opponent_idx, [])),
            'self_moves': copy.deepcopy(self.quick_action_memory_for_llm.get(current_player_idx, [])),
            'player_idx': current_player_idx,
        }
        return res
        pass

    def reset(self):
        self.game_name = 'tic_tac_toe'
        super().reset()
        self.game_name = 'tictactoe'
        pass

    def agent_action_to_openspiel(self, action):
        try:
            numbers_match = re.search(r'C(\d+)R(\d+)', action)

            if numbers_match:
                column = int(numbers_match.group(1))
                row = int(numbers_match.group(2))
            result = column-1+(row-1)*3
            return result
        except Exception:
            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None
