import numpy as np
from typing import List
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils
from gamingbench.games.openspiel_adapter import OpenSpielGame


class FirstSealedAuction(OpenSpielGame):
    def __init__(self, config=None) -> None:
        super().__init__("first_sealed_auction", config=config)
        pass

    def openspiel_action_to_agent(self, action):
        agent_action_list = [a.split(' ')[-1] for a in action]
        agent_action_list = [f'<{a}>' for a in agent_action_list]
        return agent_action_list

    def _sample_chance_action(self, action_list, prob_list):
        # Filter out 0 valuation if it exists
        filtered_actions = []
        filtered_probs = []
        for a, p in zip(action_list, prob_list):
            if a != 0:
                filtered_actions.append(a)
                filtered_probs.append(p)

        rng = getattr(self, '_rng', None)
        if filtered_actions:
            if rng is not None:
                return rng.choices(population=filtered_actions, weights=filtered_probs)[0]
            prob_sum = sum(filtered_probs)
            return np.random.choice(filtered_actions, p=[p / prob_sum for p in filtered_probs])

        if rng is not None:
            return rng.choices(population=list(action_list), weights=list(prob_list))[0]
        return np.random.choice(action_list, p=prob_list)

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        val = self.env.observation_string(current_player_idx)
        # Convert to int to prevent LLM from thinking it can bid floats like 0.5
        val_int = int(float(val))
        past_moves = self.quick_action_memory_for_llm.get(current_player_idx, [])
        return {
            'board': f"Your private valuation: {val_int}",
            'valuation': val_int,
            'self_moves': past_moves
        }

    def get_opponent_board_state(self, board_str):
        return board_str.replace('Your private valuation', 'Opponent\'s private valuation')

    def agent_action_to_openspiel(self, action):
        try:
            # Parse as float first to handle cases where LLM hallucinates <0.5>
            bid = int(round(float(action[1:-1])))
            legal_actions = self.env.legal_actions(self.env.current_player())
            legal_actions = [int(l) for l in legal_actions]
            if bid in legal_actions:
                return bid
            distance = float('inf')
            ans = bid
            for i in legal_actions:
                d = abs(i-bid)
                if d <= distance:
                    distance = d
                    ans = i
            return ans
        except Exception as e:
            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None
