import re
from typing import List, Dict, Any
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils

# Import the hanabi env; fail gracefully if not installed
try:
    from hanabi_learning_environment import rl_env
except ImportError:
    rl_env = None

class Hanabi:
    def __init__(self, config=None) -> None:
        self.game_name = "hanabi"
        self.num_players = getattr(config, 'num_players', 2) if config else 2
        self.variant = getattr(config, 'variant', 'Hanabi-Full') if config else 'Hanabi-Full'
        self.config = config
        self.game = self
        
        self.logger = utils.LLMBenchLogger(None)
        self.status = "Normal"
        
        colors = getattr(self.config, 'colors', 5) if self.config else 5
        ranks = getattr(self.config, 'ranks', 5) if self.config else 5
        self.max_score = colors * ranks

        if rl_env is None:
            raise ImportError("hanabi_learning_environment is not installed. Run: pip install /nas/longleaf/home/anvu/Avalon/hanabi-learning-environment/")
            
        self.env = self._create_env()
        self._move_log = []

    def reset(self):
        self.env = self._create_env()
        self.status = "Normal"
        self._move_log = []

    def _create_env(self):
        if self.variant in ["Hanabi-Full", "Hanabi-Full-CardKnowledge", "Hanabi-Full-Minimal", "Hanabi-Small", "Hanabi-Very-Small"]:
            return rl_env.make(self.variant, num_players=self.num_players)
            
        env_config = {
            "colors": getattr(self.config, 'colors', 5),
            "ranks": getattr(self.config, 'ranks', 5),
            "players": self.num_players,
            "hand_size": getattr(self.config, 'hand_size', 5),
            "max_information_tokens": getattr(self.config, 'max_information_tokens', 8),
            "max_life_tokens": getattr(self.config, 'max_life_tokens', 3),
            "observation_type": getattr(self.config, 'observation_type', 1)
        }
        return rl_env.HanabiEnv(config=env_config)

    def print_game_info(self):
        self.logger.info(f"Hanabi variant: {self.variant}, Players: {self.num_players}")

    def play(self, agent_list, model_list, tracker, first_player: int = 0, seat_mapping: list = None):
        self.status = "Normal"
        _match = GameMatch()
        self._move_log = []
        
        if seat_mapping is None:
            seat_mapping = [(i + first_player) % self.num_players for i in range(self.num_players)]
        
        from gamingbench.prompts.observation_prompts import construct_game_intro
        # [LTM Integration] Initialize agent state for tracking
        for i, agent in enumerate(agent_list):
            if hasattr(agent, 'reset_game_state'):
                # For N-player games (num_players > 2), main.py already initializes the keys properly.
                # We should only initialize if it hasn't been set yet (e.g., interactive tests or 2-player fallbacks).
                if not getattr(agent, 'current_opponent_key', None) and not getattr(agent, 'current_opponent_keys', None):
                    game_intro = construct_game_intro(self.game_name, enable_chat=getattr(agent, 'enable_chat', False), game_config=self.config)
                    if self.num_players == 2:
                        opponent_idx = 1 - i
                        opponent_agent = agent_list[opponent_idx]
                        if hasattr(agent, 'agent_name') and hasattr(opponent_agent, 'agent_name') and agent.agent_name == opponent_agent.agent_name:
                            opponent_name = opponent_agent.agent_name
                        else:
                            opponent_name = f"{opponent_agent.agent_name}" if len(agent_list) > 1 else "unknown"
                        agent.reset_game_state(opponent_name, game_intro)
                    else:
                        opponent_keys = []
                        for j in range(self.num_players):
                            if i != j:
                                opponent_agent = agent_list[j]
                                opponent_keys.append(f"{opponent_agent.agent_name}")
                        # Sort to avoid seat-based fragmentation
                        agent.reset_game_state("+".join(sorted(opponent_keys)), game_intro)
                agent.current_player_index = i
        
        obs = self.env.reset()
        done = False
        
        # Initialize hint rounds, draw rounds, and player hint history
        self.hint_rounds = []
        self.card_draw_rounds = []
        self.player_hint_history = []
        for i in range(self.num_players):
            hand_size = len(obs['player_observations'][i]['card_knowledge'][0])
            self.hint_rounds.append([{} for _ in range(hand_size)])
            self.card_draw_rounds.append([0 for _ in range(hand_size)])
            self.player_hint_history.append([])
            
        round_num = 1
        while not done:
            seat_idx = obs['current_player']
            logical_idx = seat_mapping[seat_idx]
            player_obs = obs['player_observations'][seat_idx]
            
            # Record state snapshot for the log
            state_snapshot = self._get_state_snapshot_str_full(obs, seat_idx, logical_idx, agent_list, model_list, seat_mapping, round_num=round_num, hint_rounds=self.hint_rounds)
            
            observation_dict = self._build_obs_dict(obs, player_obs, seat_idx, logical_idx, agent_list, model_list, seat_mapping, round_num=round_num, hint_rounds=self.hint_rounds)
            
            # Step the agent
            action_str, query_list = agent_list[logical_idx].step(observation_dict)
            
            # Parse action
            legal_moves_raw = player_obs['legal_moves']
            action_dict = self._parse_action(action_str, legal_moves_raw, logical_idx, seat_idx, seat_mapping, agent_list, model_list)
            
            if action_dict is None:
                self.logger.warning(f"Unsuccessful interpreting LLM move: {action_str}. Falling back to first legal move.")
                if not legal_moves_raw:
                    break
                action_dict = legal_moves_raw[0]
                action_str = self._action_dict_to_str(action_dict, logical_idx, seat_idx, seat_mapping, agent_list, model_list)
            
            self.logger.info(f"player: {logical_idx} agent:{agent_list[logical_idx].agent_name}, action: {action_str}")
            self.logger.info(f"game_action:{action_str}")
            
            # Step the environment
            next_obs, reward, done, info = self.env.step(action_dict)
            
            # Update hint rounds
            if not done:
                atype = action_dict['action_type']
                if atype in ['PLAY', 'DISCARD']:
                    idx = action_dict['card_index']
                    self.hint_rounds[seat_idx].pop(idx)
                    self.card_draw_rounds[seat_idx].pop(idx)
                    new_hand_size = len(next_obs['player_observations'][seat_idx]['card_knowledge'][0])
                    if len(self.hint_rounds[seat_idx]) < new_hand_size:
                        self.hint_rounds[seat_idx].append({})
                        self.card_draw_rounds[seat_idx].append(round_num)
                elif atype in ['REVEAL_COLOR', 'REVEAL_RANK']:
                    target_seat = (seat_idx + action_dict['target_offset']) % self.num_players
                    
                    if atype == 'REVEAL_COLOR':
                        self.player_hint_history[target_seat].append(f"[Round {round_num}: Color {action_dict['color']}]")
                    elif atype == 'REVEAL_RANK':
                        self.player_hint_history[target_seat].append(f"[Round {round_num}: Rank {action_dict['rank'] + 1}]")
                        
                    old_ck = obs['player_observations'][target_seat]['card_knowledge'][0]
                    new_ck = next_obs['player_observations'][target_seat]['card_knowledge'][0]
                    for c_idx in range(len(old_ck)):
                        if atype == 'REVEAL_COLOR' and old_ck[c_idx]['color'] != new_ck[c_idx]['color']:
                            self.hint_rounds[target_seat][c_idx]['color'] = round_num
                        if atype == 'REVEAL_RANK' and old_ck[c_idx]['rank'] != new_ck[c_idx]['rank']:
                            self.hint_rounds[target_seat][c_idx]['rank'] = round_num
            
            # Track step
            _step = Step(agent_list[logical_idx].agent_name)
            _step.set_model_name(model_list[logical_idx].nick_name)
            _step.set_observation(observation_dict['board'])
            _step.set_move(action_str)
            for query in query_list:
                _step.add_query(query)
            _match.add_step(_step)
            
            # Generate outcome string
            outcome_str = self._get_outcome_str(action_dict, player_obs, next_obs, reward, seat_idx, logical_idx, agent_list, model_list, seat_mapping)
            
            self._move_log.append((logical_idx, action_str, outcome_str, state_snapshot))
            
            obs = next_obs
            round_num += 1

        final_score = self.env.state.score()
        _match.winner_score = final_score
        _match.set_winner("")  # cooperative game, no individual winner
        _match.status = self.status
        tracker.add_match(_match)

        # Generate shared game history
        game_history_str = self._generate_game_history(final_score, agent_list, model_list)
        
        final_board_state = self._get_state_snapshot_str_full(obs, 0, seat_mapping[0], agent_list, model_list, seat_mapping, round_num=round_num, hint_rounds=self.hint_rounds) if 'player_observations' in obs else ""

        # Post-game updates for LTM
        for agent in agent_list:
            if hasattr(agent, 'post_game_update'):
                agent.post_game_update(game_history_str, final_board_state, env_name=self.game_name)

    def _build_obs_dict(self, obs, player_obs, seat_idx, logical_idx, agent_list, model_list, seat_mapping, round_num=None, hint_rounds=None):
        # Human readable legal moves
        legal_moves_str = [self._action_dict_to_str(a, logical_idx, seat_idx, seat_mapping, agent_list, model_list) for a in player_obs['legal_moves']]
        
        # Format fireworks
        fireworks = player_obs['fireworks']
        
        # Other hands
        other_hands = {}
        for offset, hand in enumerate(player_obs['observed_hands']):
            if offset == 0:
                continue # own hand is not visible in observed_hands in same way, actually offset 1 is next player
            # in rl_env, observed_hands has (num_players - 1) elements
            actual_seat = (seat_idx + offset) % self.num_players
            actual_logical = seat_mapping[actual_seat]
            formatted_hand = []
            
            other_ck = player_obs['card_knowledge'][offset]
            
            for c_idx, card in enumerate(hand):
                # Format card hint
                ck = other_ck[c_idx]
                hr = hint_rounds[actual_seat][c_idx] if hint_rounds else {}
                k = []
                if ck['color'] is not None:
                    round_str = f" @ Round {hr.get('color', '?')}" if hr.get('color') else ""
                    k.append(f"color={ck['color']}{round_str}")
                if ck['rank'] is not None:
                    round_str = f" @ Round {hr.get('rank', '?')}" if hr.get('rank') else ""
                    k.append(f"rank={ck['rank'] + 1}{round_str}")
                hint_str = "hints: " + (", ".join(k) if k else "none")
                
                draw_round = self.card_draw_rounds[actual_seat][c_idx]
                formatted_hand.append(f"{card['color']}{card['rank'] + 1} (Drawn Round {draw_round}) [{hint_str}]")
            
            agent = agent_list[actual_logical] if agent_list and len(agent_list) > actual_logical else None
            name = f"{agent.agent_name}" if agent and hasattr(agent, 'agent_name') and model_list else f"Player {actual_logical}"
            other_hands[name] = formatted_hand
            
        # Own hand knowledge
        own_knowledge = []
        for c_idx, card_knowledge in enumerate(player_obs['card_knowledge'][0]):
            hr = hint_rounds[seat_idx][c_idx] if hint_rounds else {}
            k = []
            if card_knowledge['color'] is not None:
                round_str = f" @ Round {hr.get('color', '?')}" if hr.get('color') else ""
                k.append(f"color={card_knowledge['color']}{round_str}")
            if card_knowledge['rank'] is not None:
                round_str = f" @ Round {hr.get('rank', '?')}" if hr.get('rank') else ""
                k.append(f"rank={card_knowledge['rank'] + 1}{round_str}")
                
            draw_round = self.card_draw_rounds[seat_idx][c_idx]
            own_knowledge.append(f"(Drawn Round {draw_round}) hints: " + (", ".join(k) if k else "none"))
            
        discard_pile = [f"{c['color']}{c['rank'] + 1}" for c in player_obs['discard_pile']]
        
        # Board string for history tracking
        board_str = self._get_state_snapshot_str_full(obs, seat_idx, logical_idx, agent_list, model_list, seat_mapping, round_num=round_num, hint_rounds=hint_rounds)
        
        hint_histories = {}
        for s in range(self.num_players):
            s_logical = seat_mapping[s]
            agent = agent_list[s_logical] if agent_list and len(agent_list) > s_logical else None
            name = f"{agent.agent_name}" if agent and hasattr(agent, 'agent_name') and model_list else f"Player {s_logical}"
            hint_histories[name] = self.player_hint_history[s]
        
        agent = agent_list[logical_idx] if agent_list and len(agent_list) > logical_idx else None
        player_name = f"{agent.agent_name}" if agent and hasattr(agent, 'agent_name') and model_list else f"Player {logical_idx}"
        
        return {
            'env_name': self.game_name,
            'player_idx': logical_idx,
            'player_name': player_name,
            'num_players': self.num_players,
            'round_num': round_num,
            'fireworks': fireworks,
            'info_tokens': player_obs['information_tokens'],
            'life_tokens': player_obs['life_tokens'],
            'deck_size': player_obs['deck_size'],
            'discard_pile': discard_pile,
            'own_card_knowledge': own_knowledge,
            'other_hands': other_hands,
            'hint_histories': hint_histories,
            'legal_moves': legal_moves_str,
            'legal_moves_raw': player_obs['legal_moves'],
            'board': board_str
        }

    def _parse_action(self, action_str, legal_moves_raw, logical_idx, seat_idx, seat_mapping, agent_list=None, model_list=None):
        if not action_str:
            return None
            
        # Clean < > 
        action_str_clean = action_str.replace('<', '').replace('>', '').strip()
        
        # Verify it's legal by directly comparing strings first
        for legal_move in legal_moves_raw:
            legal_str = self._action_dict_to_str(legal_move, logical_idx, seat_idx, seat_mapping, agent_list, model_list)
            if action_str.strip() == legal_str or action_str_clean == legal_str.replace('<', '').replace('>', ''):
                return legal_move

        # Fallback to parts parsing
        parts = action_str_clean.split(' ')
        if len(parts) == 0:
            return None
            
        action_type = parts[0].upper()
        
        try:
            if action_type in ['PLAY', 'DISCARD'] and len(parts) >= 2:
                idx = int(parts[1])
                candidate = {'action_type': action_type, 'card_index': idx}
            elif action_type == 'HINT':
                # Try to find a legal hint action that matches the color or rank in the string
                for legal_move in legal_moves_raw:
                    if legal_move['action_type'] not in ['REVEAL_COLOR', 'REVEAL_RANK']:
                        continue
                    legal_str = self._action_dict_to_str(legal_move, logical_idx, seat_idx, seat_mapping, agent_list, model_list).replace('<', '').replace('>', '')
                    if legal_str in action_str_clean:
                        return legal_move
                return None
            else:
                return None
        except ValueError:
            return None
            
        # Verify it's legal
        for legal_move in legal_moves_raw:
            match = True
            for k, v in candidate.items():
                if legal_move.get(k) != v:
                    match = False
                    break
            if match:
                return legal_move
                
        return None

    def _action_dict_to_str(self, action_dict, logical_idx, seat_idx, seat_mapping, agent_list=None, model_list=None):
        atype = action_dict['action_type']
        if atype in ['PLAY', 'DISCARD']:
            return f"<{atype} {action_dict['card_index']}>"
        elif atype in ['REVEAL_COLOR', 'REVEAL_RANK']:
            target_seat = (seat_idx + action_dict['target_offset']) % self.num_players
            player = seat_mapping[target_seat]
            if agent_list and model_list:
                agent = agent_list[player]
                name = f"{agent.agent_name}" if hasattr(agent, 'agent_name') else f"Player {player}"
            else:
                name = f"PLAYER {player}"
                
            if atype == 'REVEAL_COLOR':
                return f"<HINT {name} COLOR {action_dict['color']}>"
            else:
                return f"<HINT {name} RANK {action_dict['rank'] + 1}>"
        return "<UNKNOWN>"

    def _get_state_snapshot_str_full(self, obs, seat_idx, logical_idx, agent_list, model_list, seat_mapping, round_num=None, hint_rounds=None):
        if not obs or 'player_observations' not in obs:
            return ""
            
        p_obs = obs['player_observations'][seat_idx]
        fw = p_obs['fireworks']
        fw_str = ",".join([f"{k}={v}" for k, v in fw.items()])
        
        discard_pile_str = ",".join([f"{c['color']}{c['rank'] + 1}" for c in p_obs['discard_pile']])
        base_state = f"Fireworks: {fw_str} | Life: {p_obs['life_tokens']} | Info: {p_obs['information_tokens']} | Deck: {p_obs['deck_size']} | Discard: {discard_pile_str}"
        
        hands_str = []
        for s in range(self.num_players):
            offset = (s - seat_idx + self.num_players) % self.num_players
            actual_hand = p_obs['observed_hands'][offset]
            knowledge = p_obs['card_knowledge'][offset]
            
            hand_parts = []
            for c_idx, (card, k) in enumerate(zip(actual_hand, knowledge)):
                k_list = []
                hr = hint_rounds[s][c_idx] if hint_rounds else {}
                if k['color'] is not None:
                    round_str = f" @ Round {hr.get('color', '?')}" if hr.get('color') else ""
                    k_list.append(f"color={k['color']}{round_str}")
                if k['rank'] is not None:
                    round_str = f" @ Round {hr.get('rank', '?')}" if hr.get('rank') else ""
                    k_list.append(f"rank={k['rank'] + 1}{round_str}")
                hint_str = "hints: " + (", ".join(k_list) if k_list else "none")
                
                if card['color'] is None and card['rank'] == -1:
                    card_str = "Unknown"
                else:
                    card_str = f"{card['color']}{card['rank'] + 1}"
                    
                draw_round = self.card_draw_rounds[s][c_idx]
                hand_parts.append(f"{card_str} (Drawn Round {draw_round}) [{hint_str}]")
                
            l_idx = seat_mapping[s]
            agent = agent_list[l_idx] if agent_list and len(agent_list) > l_idx else None
            if agent and hasattr(agent, 'agent_name') and model_list:
                name = f"{agent.agent_name}"
            else:
                name = f"Player {l_idx}"
            prefix = f"{name} (acting)" if s == seat_idx else f"{name}"
            hands_str.append(f"{prefix}: " + ", ".join(hand_parts))
            
        return base_state + "\n    [Hands] " + " | ".join(hands_str)

    def _get_outcome_str(self, action_dict, obs, next_obs, reward, seat_idx, logical_idx, agent_list, model_list, seat_mapping):
        # Provide a human-readable outcome of the action
        atype = action_dict['action_type']
        if atype == 'PLAY':
            if reward > 0:
                return f"SUCCESS: Card played. Score: {self.env.state.score()}/{self.max_score}"
            else:
                return f"FAIL: Invalid play. Life: {obs['life_tokens']}->{next_obs['player_observations'][0]['life_tokens']}"
        elif atype == 'DISCARD':
            return f"Card discarded. Info tokens: {obs['information_tokens']}->{next_obs['player_observations'][0]['information_tokens']}"
        elif atype in ['REVEAL_COLOR', 'REVEAL_RANK']:
            # Find which cards matched
            target_seat = (seat_idx + action_dict['target_offset']) % self.num_players
            target_logical = seat_mapping[target_seat]
            agent = agent_list[target_logical] if agent_list and len(agent_list) > target_logical else None
            if agent and hasattr(agent, 'agent_name') and model_list:
                name = f"{agent.agent_name}"
            else:
                name = f"Player {target_logical}"
            return f"Hint given to {name}."
        return ""

    def _generate_game_history(self, final_score, agent_list, model_list):
        history = []
        history.append(f"[Game Context] Cooperative Hanabi ({self.variant}).")
        
        teammates = []
        for i in range(len(agent_list)):
            agent = agent_list[i]
            if hasattr(agent, 'agent_name') and model_list:
                name = f"{agent.agent_name}"
            else:
                name = f"Player {i}"
            teammates.append(f"{name}")
        history.append(f"[Teammates]: {', '.join(teammates)}")
        history.append("Note: For HINT actions, use the exact target teammate name listed above to target the specific player.")
        history.append("")
        
        for i, (logical_idx, action_str, outcome_str, state_snapshot) in enumerate(self._move_log):
            history.append(f"Round {i+1}:")
            history.append(f"  [State] {state_snapshot}")
            agent = agent_list[logical_idx] if agent_list and len(agent_list) > logical_idx else None
            if agent and hasattr(agent, 'agent_name') and model_list:
                name = f"{agent.agent_name}"
            else:
                name = f"Player {logical_idx}"
            history.append(f"  [Move] {name}: {action_str}  -> {outcome_str}")
            history.append("")
            
        history.append(f"Game Outcome: Cooperative final score = {final_score}.")
        return "\n".join(history)
