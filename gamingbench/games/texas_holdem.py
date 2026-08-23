import rlcard
import re
from typing import List
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils
from gamingbench.chat.chat_channel import ChatChannel

class TexasHoldem:
    def __init__(self, config=None) -> None:
        self.game_name = "texas_holdem"
        self.num_players = getattr(config, 'num_players', 2) if config else 2
        self.config = config
        self.game = self
        
        self.logger = utils.LLMBenchLogger(None)
        self.status = "Normal"
        self.env = rlcard.make('limit-holdem', config={'game_num_players': self.num_players})
        self._move_log = []
        self.quick_action_memory_for_llm = {}
        self.action_record = ["Preflop"]

    def reset(self):
        self.env = rlcard.make('limit-holdem', config={'game_num_players': self.num_players})
        self.status = "Normal"
        self._move_log = []
        self.quick_action_memory_for_llm = {}
        self.action_record = ["Preflop"]

    def print_game_info(self):
        self.logger.info(f"Texas Hold'em Players: {self.num_players}")

    def play(self, agent_list, model_list, tracker, first_player: int = 0, seat_mapping: list = None):
        self.status = "Normal"
        _match = GameMatch()
        self._move_log = []
        self.action_record = ["Preflop"]
        chat_channel = ChatChannel(window_size=4)
        
        if seat_mapping is None:
            seat_mapping = [(i + first_player) % self.num_players for i in range(self.num_players)]
        
        from gamingbench.prompts.observation_prompts import construct_game_intro
        for i, agent in enumerate(agent_list):
            if hasattr(agent, 'reset_game_state'):
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
                        agent.reset_game_state("+".join(sorted(opponent_keys)), game_intro)
                agent.current_player_index = i
        
        state, current_player = self.env.reset()
        
        if self.num_players == 2:
            raw_chips = state['raw_obs']['all_chips']
            if raw_chips[0] < raw_chips[1]:
                self.sb_seat = 0
            else:
                self.sb_seat = 1
        else:
            self.sb_seat = -1
            
        num_step = 0
        
        while not self.env.is_over():
            seat_idx = current_player
            logical_idx = seat_mapping[seat_idx]
            
            # Chat Phase
            if all(getattr(agent, "enable_chat", False) for agent in agent_list):
                # 1. Active player speaks
                obs_dict_active = self._build_obs_dict(state, logical_idx, agent_list, model_list, seat_mapping, chat_phase=True, active_player=True, round_num=num_step+1)
                chat_history_active = chat_channel.get_recent_window(logical_idx)
                msg_active, _ = agent_list[logical_idx].chat_step(obs_dict_active, chat_history_active)
                if msg_active:
                    current_round = (num_step // self.num_players) + 1
                    chat_channel.add_message(logical_idx, msg_active, round_idx=current_round)
                    
                # 2. Peer player speaks (disabled for Agent-Pro style Texas Holdem)
                pass
            
            # Action Phase
            observation_dict = self._build_obs_dict(state, logical_idx, agent_list, model_list, seat_mapping, round_num=num_step+1)
            observation_dict['chat_context'] = chat_channel.get_recent_window(logical_idx) if all(getattr(a, "enable_chat", False) for a in agent_list) else ""
            
            action_str, query_list = agent_list[logical_idx].step(observation_dict)
            
            legal_moves_raw = state['raw_legal_actions']
            action_id = self._parse_action(action_str, legal_moves_raw)
            
            if action_id is None:
                self.logger.warning(f"Unsuccessful interpreting LLM move: {action_str}. Folding by default.")
                action_str = "fold" if "fold" in legal_moves_raw else legal_moves_raw[0]
                action_id = self.env.actions.index(action_str)
                
            self.logger.info(f"player: {logical_idx} agent:{agent_list[logical_idx].agent_name}, action: {action_str}")
            
            act = self.quick_action_memory_for_llm.get(logical_idx, [])
            act.append(action_str)
            self.quick_action_memory_for_llm[logical_idx] = act
            
            _step = Step(agent_list[logical_idx].agent_name)
            _step.set_model_name(model_list[logical_idx].nick_name)
            _step.set_observation(observation_dict['board_str'])
            _step.set_move(action_str)
            for query in query_list:
                _step.add_query(query)
            _match.add_step(_step)
            
            board_state = observation_dict['history_pos']
            self._move_log.append((logical_idx, action_str, board_state))
            
            self.action_record.append(f"Player {logical_idx} {action_str}s")
            
            state, current_player = self.env.step(action_id)
            num_step += 1
            
            if not self.env.is_over():
                stage_map = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}
                k = len(state['raw_obs']['public_cards'])
                if stage_map.get(k) and stage_map[k] not in self.action_record:
                    self.action_record.append(stage_map[k])

        raw_payoffs = self.env.get_payoffs()
        payoffs = [float(p) for p in raw_payoffs]
        
        # Determine winner
        if payoffs[seat_mapping.index(0)] > payoffs[seat_mapping.index(1)]:
            winner_name = agent_list[0].agent_name
            _match.winner_score = payoffs[seat_mapping.index(0)]
            _match.loser_score = payoffs[seat_mapping.index(1)]
        elif payoffs[seat_mapping.index(1)] > payoffs[seat_mapping.index(0)]:
            winner_name = agent_list[1].agent_name
            _match.winner_score = payoffs[seat_mapping.index(1)]
            _match.loser_score = payoffs[seat_mapping.index(0)]
        else:
            winner_name = ""
            _match.winner_score = payoffs[seat_mapping.index(0)]
            _match.loser_score = payoffs[seat_mapping.index(1)]
            
        _match.set_winner(winner_name)
        _match.status = self.status
        tracker.add_match(_match)
        
        if _match.winner != "":
            self.logger.info(f"The winner is {_match.winner}")
        else:
            self.logger.info("There are no winner in this game.")

        # Post-game updates for LTM
        for agent_idx, agent in enumerate(agent_list):
            if hasattr(agent, 'post_game_update'):
                agent_seat = seat_mapping.index(agent_idx)
                game_history_str = self._generate_game_history(agent_idx, agent_seat, payoffs, seat_mapping, agent_list, model_list)
                final_state = self.env.get_state(agent_seat)
                final_obs_dict = self._build_obs_dict(final_state, agent_idx, agent_list, model_list, seat_mapping, round_num=num_step+1)
                final_board_state = final_obs_dict['board_str']
                
                try:
                    agent.post_game_update(game_history_str, final_board_state=final_board_state, env_name=self.game_name)
                except TypeError:
                    try:
                        agent.post_game_update(game_history_str, final_board_state=final_board_state)
                    except TypeError:
                        agent.post_game_update(game_history_str)

    def _build_obs_dict(self, state, logical_idx, agent_list, model_list, seat_mapping, chat_phase=False, active_player=True, round_num=None):
        raw_obs = state['raw_obs']
        legal_moves_raw = state['raw_legal_actions']
        legal_moves_str = legal_moves_raw
        
        try:
            my_seat = seat_mapping.index(logical_idx)
        except ValueError:
            my_seat = 0

        total_pot = sum(raw_obs['all_chips'])
        max_bid = max(raw_obs['all_chips'])
        my_bid = raw_obs['all_chips'][my_seat]
        cost_to_call = max_bid - my_bid
        
        if cost_to_call == 0:
            call_action_msg = "0 chips (Checking is legal)"
        else:
            call_action_msg = f"{cost_to_call} chip(s) (Facing a bet/raise)"

        if self.num_players == 2:
            opp_seat = 1 - my_seat
            opp_bid = raw_obs['all_chips'][opp_seat]
            chips_str = f"Total Pot: {total_pot} chips. Your total investment: {my_bid}, Opponent's total investment: {opp_bid}. Cost to Call: {call_action_msg}"
            
            if getattr(self, 'sb_seat', -1) == my_seat:
                role_str = "Small Blind (acts first preflop, second postflop)"
            else:
                role_str = "Big Blind (acts second preflop, first postflop)"
            role_intro = f"You are the {role_str}. "
        else:
            chips_str = f"Total Pot: {total_pot} chips. Chips invested by each player: {raw_obs['all_chips']}, meaning your total investment is {my_bid} chips. Cost to Call: {call_action_msg}"
            role_intro = ""

        board_str = f"{role_intro}Now your hand is {raw_obs['hand']}, and the community cards is {raw_obs['public_cards']}. {chips_str}"

        stage_map = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}
        k = len(raw_obs['public_cards'])
        stage_str = stage_map.get(k, "Unknown")
        history_pos = f"Stage: {stage_str}, Hand: {raw_obs['hand']}, Community: {raw_obs['public_cards']}, {chips_str}"

        return {
            'env_name': self.game_name,
            'player_idx': logical_idx,
            'num_players': self.num_players,
            'round_num': round_num,
            'hand': raw_obs['hand'],
            'public_cards': raw_obs['public_cards'],
            'all_chips': raw_obs['all_chips'],
            'my_chips': raw_obs['my_chips'],
            'pot': raw_obs.get('pot', 0),
            'stakes': raw_obs.get('stakes', []),
            'stage': raw_obs.get('stage', 0),
            'legal_moves': legal_moves_str,
            'legal_moves_raw': legal_moves_raw,
            'board_str': board_str,
            'history_pos': history_pos,
            'is_chat_phase': chat_phase,
            'is_active_player': active_player,
            'action_record': self._format_action_record(seat_mapping, agent_list, model_list)
        }

    def _parse_action(self, action_str, legal_moves_raw):
        if not action_str:
            return None
        try:
            import json, re
            match = re.search(r"{\s*'action'\s*:\s*'(\w+)'\s*}", action_str)
            if match:
                parsed_str = match.group(1).lower()
            else:
                parsed_str = action_str.strip().lower()
        except:
            parsed_str = action_str.strip().lower()
            
        if parsed_str in legal_moves_raw:
            return self.env.actions.index(parsed_str)
        return None

    def _format_action_record(self, seat_mapping=None, agent_list=None, model_list=None):
        result = "\n--- Game History ---\n"
        current_round = "PREFLOP"
        for item in self.action_record:
            if item in ["Preflop", "Flop", "Turn", "River"]:
                current_round = item.upper()
                if item == "Preflop":
                    result += "** Preflop **\n"
                elif item == "Flop":
                    result += "** Flop ** (3 Community Cards)\n"
                elif item == "Turn":
                    result += "** Turn ** (4 Community Cards)\n"
                elif item == "River":
                    result += "** River ** (5 Community Cards)\n"
            else:
                formatted_item = item
                if seat_mapping and self.num_players == 2:
                    match = re.search(r"Player (\d+)", item)
                    if match:
                        pid = int(match.group(1))
                        try:
                            seat_idx = seat_mapping.index(pid)
                            role = "SB" if getattr(self, 'sb_seat', -1) == seat_idx else "BB"
                            if agent_list and model_list and len(agent_list) > pid:
                                agent = agent_list[pid]
                                if hasattr(agent, 'agent_name'):
                                    agent_display_name = f"{agent.agent_name}"
                                else:
                                    agent_display_name = f"Player {pid}"
                            else:
                                agent_display_name = f"Player {pid}"
                            formatted_item = item.replace(f"Player {pid}", f"{agent_display_name} ({role})")
                        except ValueError:
                            pass
                result += f"- {formatted_item}\n"
        result += "--------------------\n"
        result += f"Current Round: {current_round}."
        return result

    def get_opponent_board_state(self, board_str):
        return board_str.replace('Hand:', 'Opponent Hand:')

    def _generate_game_history(self, agent_idx, agent_seat, payoffs, seat_mapping, agent_list=None, model_list=None):
        history = []
        if agent_list and model_list and len(agent_list) > agent_idx:
            agent = agent_list[agent_idx]
            opp_idx = 1 - agent_idx if self.num_players == 2 else 0
            opp_agent = agent_list[opp_idx] if self.num_players == 2 else None
            
            if hasattr(agent, 'agent_name'):
                my_name = f"{agent.agent_name}"
            else:
                my_name = f"Player {agent_idx+1}"
                
            if opp_agent and hasattr(opp_agent, 'agent_name'):
                opp_name = f"{opp_agent.agent_name}"
            elif opp_agent:
                opp_name = f"Player {opp_idx+1}"
            else:
                opp_name = "Opponents"
        else:
            my_name = f"Player {agent_idx+1}"
            opp_name = f"Player {2 if agent_idx == 0 else 1}"
            
        history.append(f"[Player Context] You play as {my_name}. The opponent plays as {opp_name}.")
        history.append(f"[Position Legend] Each [Position] line shows the game state before that player's move. Format: Stage: [stage], Hand: [hand], Community: [cards], Total Pot: [chips], Your total investment: [chips], Opponent's total investment: [chips], Cost to Call: [cost]. (Note: 'Hand' will be labeled as 'Your Hand' or 'Opponent's Hand' depending on the turn)\n")
        
        for step_idx, (logical_idx, action_str, board_state) in enumerate(self._move_log):
            current_round = step_idx + 1
            prefix = "You" if logical_idx == agent_idx else "Opponent"
            
            if prefix == "You":
                history.append(f"Round {current_round} (Your move):")
                board_state_formatted = board_state.replace('Hand:', 'Your Hand:')
                history.append(f"  [Position]: {board_state_formatted}")
            else:
                history.append(f"Round {current_round} (Opponent's move):")
                board_state_formatted = board_state.replace('Hand:', "Opponent's Hand:")
                if self.num_players == 2:
                    import re
                    match = re.search(r"Your total investment:\s*([\d\.]+),\s*Opponent's total investment:\s*([\d\.]+)", board_state_formatted)
                    if match:
                        opp_bid = match.group(1)
                        my_bid = match.group(2)
                        board_state_formatted = board_state_formatted[:match.start()] + f"Your total investment: {my_bid}, Opponent's total investment: {opp_bid}" + board_state_formatted[match.end():]
                else:
                    # Replace "meaning you have invested" with "meaning they have invested"
                    board_state_formatted = board_state_formatted.replace("meaning you have invested", "meaning they have invested")
                history.append(f"  [Position]: {board_state_formatted}")
            
            history.append(f"  [Move] {prefix}: <{action_str}>\n")
            
        your_score = payoffs[agent_seat]
        opp_score = payoffs[seat_mapping.index(1 - agent_idx)] if self.num_players == 2 else payoffs[seat_mapping.index(0)] # simplify for 2 players
        history.append(f"Game Outcome: Your net chips={your_score}, Opponent net chips={opp_score}")
        
        return "\n".join(history)
