
from gamingbench.games.openspiel_adapter import OpenSpielGame

class PrisonersDilemma(OpenSpielGame):


    def __init__(self, config=None) -> None:
        super().__init__("python_iterated_prisoners_dilemma")

    def openspiel_action_to_agent(self, action):
        return ['<Silent>', '<Testify>']

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        moves = str(self.env).strip().split(' ')
        self_moves = moves[current_player_idx].split(':')[1]
        opponent_moves = moves[0 if current_player_idx == 1 else 1].split(':')[1]
        
        board_parts = []
        for round_idx, (s_m, o_m) in enumerate(zip(self_moves, opponent_moves)):
            s_act = "Silent" if s_m == 'C' else "Testify"
            o_act = "Silent" if o_m == 'C' else "Testify"
            board_parts.append(f"Round {round_idx+1}: You={s_act}, Opponent={o_act}")
        board_str = "; ".join(board_parts) if board_parts else "No moves yet"
        
        return {
            'board': board_str,
            'self_moves': self_moves,
            'opponent_moves': opponent_moves
        }

    def agent_action_to_openspiel(self, action):
        if action == '<Silent>':
            return 0
        elif action == '<Testify>':
            return 1
        else:
            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None



