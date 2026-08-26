import numpy as np
from typing import List
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils
from gamingbench.games.openspiel_adapter import OpenSpielGame
import re


class Negotiation(OpenSpielGame):
    def __init__(self, config=None) -> None:
        super().__init__("negotiation", config=config)
        self._init_custom_utils()

    def _init_custom_utils(self):
        rng = getattr(self, '_rng', None)
        if rng is None:
            # No isolated RNG available; custom_agent_utils will be injected
            # by run_match from the pregenerated game_state. Just ensure the
            # attribute exists so downstream code never hits AttributeError.
            if not hasattr(self, 'custom_agent_utils'):
                self.custom_agent_utils = {}
            return
        self.custom_agent_utils = {}
        for p in range(2):
            cuts = sorted(rng.sample(range(1, 20), 2))
            self.custom_agent_utils[p] = [cuts[0], cuts[1] - cuts[0], 20 - cuts[1]]

    def reset(self):
        super().reset()
        # Only regenerate if an isolated RNG is present (i.e., pregeneration
        # context). In normal match play, custom_agent_utils is set directly
        # by run_match from the pregenerated game_state.
        if getattr(self, '_rng', None) is not None:
            self._init_custom_utils()


    def get_opponent_board_state(self, board_str):
        # Safely swap Opponent and Your prefixes
        board_str = board_str.replace('Opponent Proposal', '___TEMP_OPP_PROP___')
        board_str = board_str.replace('Opponent Utterance', '___TEMP_OPP_UTT___')
        board_str = board_str.replace('Your Proposal', '___TEMP_YOUR_PROP___')
        board_str = board_str.replace('Your Utterance', '___TEMP_YOUR_UTT___')
        
        board_str = board_str.replace('Your values', 'Opponent\'s values')
        
        board_str = board_str.replace('___TEMP_OPP_PROP___', 'Your Proposal')
        board_str = board_str.replace('___TEMP_OPP_UTT___', 'Your Utterance')
        board_str = board_str.replace('___TEMP_YOUR_PROP___', 'Opponent Proposal')
        board_str = board_str.replace('___TEMP_YOUR_UTT___', 'Opponent Utterance')
        return board_str

    def openspiel_action_to_agent(self, action):
        turn_type = self.get_turn_type()
        if turn_type == 'Proposal':
            agent_actions = []
            # TODO: default item pool is [5, 5, 5]
            for a in range(6):
                for b in range(6):
                    for c in range(6):
                        agent_actions.append(f'<Proposal: [{a}, {b}, {c}]>')
            agent_actions.append(f'<Agree>')

        elif turn_type == 'Utterance':
            agent_actions = []
            # TODO: default item pool is [5, 5, 5]
            for a in range(5):
                for b in range(5):
                    for c in range(5):
                        agent_actions.append(f'<Utterance: [{a}, {b}, {c}]>')

        else:
            raise ValueError()

        return agent_actions

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        opponent_idx = 1 if current_player_idx == 0 else 0
        turn_type = self.get_turn_type()

        agent_util_vec_match = re.search(r'Agent 0 util vec: (\d+ \d+ \d+)', openspiel_obs) if current_player_idx == 0 else re.search(
            r'Agent 1 util vec: (\d+ \d+ \d+)', openspiel_obs)
        agent_util_vec = agent_util_vec_match.group(1)
        agent_util_vec = agent_util_vec.split(' ')
        agent_util_vec = [int(v) for v in agent_util_vec]

        if hasattr(self, 'custom_agent_utils') and current_player_idx in self.custom_agent_utils:
            agent_util_vec = self.custom_agent_utils[current_player_idx]

        item_pool_match = re.search(r'Item pool: (\d+ \d+ \d+)', openspiel_obs)
        item_pool = item_pool_match.group(1)
        item_pool = item_pool.split(' ')
        item_pool = [int(v) for v in item_pool]

        most_recent_proposal_match = re.search(
            r'Most recent proposal: (.+)', self.env.observation_string())
        most_recent_utterance_match = re.search(
            r'Most recent utterance: (.+)', self.env.observation_string())

        recent_proposal = most_recent_proposal_match.group(1) if most_recent_proposal_match else "None"
        recent_utterance = most_recent_utterance_match.group(1) if most_recent_utterance_match else "None"
        
        if turn_type == 'Utterance':
            # If we are in the Utterance stage, the most recent proposal was just made by US.
            board_str = f"Pool: {item_pool}, Your values: {agent_util_vec}, Stage: {turn_type}, Your Proposal: {recent_proposal}, Opponent Utterance: {recent_utterance}"
        else:
            # If we are in the Proposal stage, the most recent proposal was made by the OPPONENT.
            board_str = f"Pool: {item_pool}, Your values: {agent_util_vec}, Stage: {turn_type}, Opponent Proposal: {recent_proposal}, Opponent Utterance: {recent_utterance}"

        res = {
            'board': board_str,
            'opponent_moves': self.quick_action_memory_for_llm.get(opponent_idx, []),
            'self_moves': self.quick_action_memory_for_llm.get(current_player_idx, []),
            'turn_type': turn_type,
            'self_value_vector': agent_util_vec,
            'item_pool': item_pool,
            'most_recent_proposal': most_recent_proposal_match.group(1)[1:-1].replace(',', '').split(' ') if most_recent_proposal_match is not None else None,
            'most_recent_utterance': most_recent_utterance_match.group(1)[1:-1].replace(',', '').split(' ') if most_recent_utterance_match is not None else None
        }
        return res
        pass

    def get_turn_type(self):
        turn_type_match = re.search(r'Turn Type: (\w+)', str(self.env))
        turn_type = turn_type_match.group(1)
        return turn_type

    def encode_integer(self, container, num_digit_values):
        encoded_value = 0
        for digit in container:
            encoded_value = encoded_value * num_digit_values + digit
        return encoded_value

    def agent_action_to_openspiel(self, action):
        try:
            numbers_match = re.search(r'\[(\d+), (\d+), (\d+)\]', action)
            print(f"debug : numbers_match:{numbers_match},action:{action}")
            if self.get_turn_type() == 'Proposal':
                if action.lower().__contains__('agree'):
                    player_idx = self.env.current_player()
                    return self.env.legal_actions(player_idx)[-1]
                else:

                    first_number = int(numbers_match.group(1))
                    second_number = int(numbers_match.group(2))
                    third_number = int(numbers_match.group(3))
                    action = [min(5, first_number), min(
                        5, second_number), min(5, third_number)]
                    return self.encode_integer(action, 6)
            else:
                # kDefaultNumItems=3
                # kMaxQuantity=5
                # kDefaultNumSymbols=5
                first_number = int(numbers_match.group(1))
                second_number = int(numbers_match.group(2))
                third_number = int(numbers_match.group(3))
                action = [min(4, first_number), min(
                    4, second_number), min(4, third_number)]
                return int(pow(6, 3)) + 1 + self.encode_integer(action, 5)
        except Exception as e:
            self.logger.error(e)
            self.logger.info("Unsuccessful interpreting LLM move")
            self.logger.info(action)
            return None

    def get_returns(self):
        orig_returns = self.env.returns()
        if orig_returns[0] == 0.0 and orig_returns[1] == 0.0:
            return orig_returns
        
        history = self.env.history()
        last_proposal_action = None
        proposer_idx = None
        
        temp_state = self.game.new_initial_state()
        for act in history:
            if act < 216 and not temp_state.is_chance_node():
                last_proposal_action = act
                proposer_idx = temp_state.current_player()
            temp_state.apply_action(act)
            
        if last_proposal_action is None:
            return orig_returns
            
        a = last_proposal_action // 36
        b = (last_proposal_action % 36) // 6
        c = last_proposal_action % 6
        proposal = [a, b, c]
        
        pool = [5, 5, 5]
        p0_items = proposal if proposer_idx == 0 else [pool[i] - proposal[i] for i in range(3)]
        p1_items = proposal if proposer_idx == 1 else [pool[i] - proposal[i] for i in range(3)]
        
        p0_utils = self.custom_agent_utils[0]
        p1_utils = self.custom_agent_utils[1]
        
        score0 = sum(p0_items[i] * p0_utils[i] for i in range(3))
        score1 = sum(p1_items[i] * p1_utils[i] for i in range(3))
        
        return [float(score0), float(score1)]
