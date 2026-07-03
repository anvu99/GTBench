

from gamingbench.games.openspiel_adapter import OpenSpielGame


class KuhnPoker(OpenSpielGame):

    def __init__(self) -> None:
        super().__init__("kuhn_poker")
        pass

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):

        state = str(self.env).split(' ')
        print(state)
        card_mapping = {'0': 'Jack (J)', '1': 'Queen (Q)', '2': 'King (K)'}
        my_card = card_mapping.get(state[current_player_idx], state[current_player_idx])
        moves_list = state[-1] if (state[-1] != '0' and state[-1] != '1' and state[-1] != '2') else None
        
        if moves_list:
            formatted_moves = []
            for idx, m in enumerate(moves_list):
                action_player_idx = 0 if idx % 2 == 0 else 1
                role = "You" if action_player_idx == current_player_idx else "Opponent"
                move_name = "Bet" if m == 'b' else "Pass"
                formatted_moves.append(f"{role}:{move_name}")
            moves_str = ", ".join(formatted_moves)
        else:
            moves_str = "No moves yet"
            
        board_str = f"Your card: {my_card}. Betting history: {moves_str}"
        
        observations = {
            'board': board_str,
            'card': state[current_player_idx],
            'moves': moves_list,
            'player_idx': current_player_idx
        }
        return observations

    def openspiel_action_to_agent(self, action):
        return [f'<{a}>' for a in action]

    def agent_action_to_openspiel(self, action):
        if action == '<Pass>':
            return 0
        elif action == '<Bet>':
            return 1
        else:
            # TODO: illegal action
            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None
