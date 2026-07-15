import numpy as np
import pyspiel
import open_spiel

from typing import List
from gamingbench.utils.history_tracker import GameMatch, Step
from gamingbench.utils import utils

from open_spiel.python import games  # import prisoners_dilemma
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.prompts.observation_prompts import construct_game_intro
from gamingbench.chat.chat_channel import ChatChannel


import copy


class OpenSpielGame:
    def __init__(self, game_name, config=None) -> None:
        self.game_name = game_name
        self.config = config
        self.game = pyspiel.load_game(game_name)
        self.env = self.game.new_initial_state()
        self.logger = utils.LLMBenchLogger(None)
        self.status = "Normal"
        self.quick_action_memory_for_llm = {}
        pass

    def reset(self):
        self.game = pyspiel.load_game(self.game_name)
        self.env = self.game.new_initial_state()
        self.logger = utils.LLMBenchLogger(None)
        self.status = "Normal"
        self.quick_action_memory_for_llm = {}

    def print_game_info(self):
        self.logger.info(self.env.agents)
        self.logger.info(self.env.agent_selection)
        self.logger.info(self.env.action_spaces)

    def play(self, agent_list, model_list, tracker):
        self.status = "Normal"
        _match = GameMatch()
        chat_channel = ChatChannel()

        # [LTM Integration] Initialize agent state for tracking
        for i, agent in enumerate(agent_list):
            if hasattr(agent, 'reset_game_state'):
                game_intro = construct_game_intro(self.game_name, enable_chat=getattr(agent, 'enable_chat', False), game_config=self.config)
                opponent_idx = 1 - i if len(agent_list) == 2 else 0
                opponent_agent = agent_list[opponent_idx]
                if hasattr(agent, 'agent_name') and hasattr(opponent_agent, 'agent_name') and agent.agent_name == opponent_agent.agent_name:
                    opponent_name = opponent_agent.agent_name
                else:
                    opponent_name = f"{opponent_agent.agent_name}_{model_list[opponent_idx].nick_name}" if len(agent_list) > 1 else "unknown"
                agent.reset_game_state(opponent_name, game_intro)

        num_step = 0
        while not self.env.is_terminal():
            if self.env.is_chance_node():
                # Chance node: sample an outcome
                outcomes = self.env.chance_outcomes()
                num_actions = len(outcomes)
                print("Chance node, got " + str(num_actions) + " outcomes")
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                print("Sampled outcome: ",
                      self.env.action_to_string(self.env.current_player(), action))
                self.env.apply_action(action)

            elif self.env.is_simultaneous_node():
                # Chat Phase (Simultaneous: Speakers rotate each round)
                if all(getattr(agent, "enable_chat", False) for agent in agent_list):
                    round_idx = num_step // self.env.num_players()
                    for i in range(self.env.num_players()):
                        player_idx = (round_idx + i) % self.env.num_players()
                        obs_dict = self.openspiel_observation_to_dict(player_idx, str(self.env))
                        obs_dict['env_name'] = self.game_name
                        
                        legal_actions = self.env.legal_actions(player_idx)
                        obs_dict['openspiel_legal_actions'] = legal_actions
                        valid_action = [self.env.action_to_string(a) for a in legal_actions]
                        obs_dict['legal_moves'] = self.openspiel_action_to_agent(valid_action)

                        chat_history = chat_channel.get_recent_window(player_idx)
                        msg, _ = agent_list[player_idx].chat_step(obs_dict, chat_history)
                        if msg:
                            current_round = (num_step // self.env.num_players()) + 1
                            chat_channel.add_message(player_idx, msg, round_idx=current_round)
                            
                # TODO: only support prisoners dilemma
                chosen_actions = []
                abnormal = False
                for player_idx in range(self.env.num_players()):
                    observation_dict = self.openspiel_observation_to_dict(
                        player_idx, str(self.env))
                    _step = Step(agent_list[player_idx].agent_name)
                    _step.set_model_name(model_list[player_idx].nick_name)
                    _step.set_observation(observation_dict)
                    legal_actions = self.env.legal_actions(player_idx)
                    observation_dict['openspiel_legal_actions'] = legal_actions
                    valid_action = [self.env.action_to_string(
                        a) for a in legal_actions]
                    valid_action = self.openspiel_action_to_agent(valid_action)
                    observation_dict['legal_moves'] = valid_action
                    observation_dict['env_name'] = self.game_name
                    observation_dict['player_idx'] = player_idx
                    observation_dict['chat_context'] = chat_channel.get_recent_window(player_idx) if all(getattr(a, "enable_chat", False) for a in agent_list) else ""
                    
                    self.logger.info(
                        f"openspiel_game_legal_action:{legal_actions}")
                    self.logger.info(f"validMove:{valid_action}")
                    action, query_list = agent_list[player_idx].step(
                        observation_dict)
                    self.logger.info(
                        f"player: {player_idx} agent:{agent_list[player_idx].agent_name}, action: {action}")
                    act = self.quick_action_memory_for_llm.get(
                        player_idx, [])

                    act.append(action)
                    self.quick_action_memory_for_llm[player_idx] = act

                    for q in query_list:
                        _step.add_query(q)

                    _step.set_move(action)
                    _match.add_step(_step)
                    game_action = self.agent_action_to_openspiel(action)
                    self.logger.info(f"game_action:{game_action}")

                    num_step += 1

                    if not self.is_valid_move(game_action, legal_actions):
                        game_action = None
                        agent_name = agent_list[player_idx].agent_name
                        self.logger.info(
                            f"agent {agent_name} made a invalid step")
                        _match.agents_at_fault.append(agent_name)
                        _match.status = "Abnormal"
                        self.status = "Abnormal"
                        abnormal = True
                        break
                    chosen_actions.append(game_action)
                if abnormal:
                    break
                self.env.apply_actions(chosen_actions)

                # inform other opponents
                for action_idx, action in enumerate(chosen_actions):
                    for player_idx, agent in enumerate(agent_list):
                        if player_idx != action_idx:
                            agent.inform_action(
                                self.env, self.env.current_player, action)

            else:
                player_idx = self.env.current_player()
                
                # Chat Phase (Sequential: Active player speaks first)
                if all(getattr(agent, "enable_chat", False) for agent in agent_list):
                    # 1. Active player speaks
                    obs_dict_active = self.openspiel_observation_to_dict(player_idx, str(self.env))
                    obs_dict_active['env_name'] = self.game_name

                    legal_actions_active = self.env.legal_actions(player_idx)
                    obs_dict_active['openspiel_legal_actions'] = legal_actions_active
                    valid_action_active = [self.env.action_to_string(a) for a in legal_actions_active]
                    obs_dict_active['legal_moves'] = self.openspiel_action_to_agent(valid_action_active)

                    chat_history_active = chat_channel.get_recent_window(player_idx)
                    msg_active, _ = agent_list[player_idx].chat_step(obs_dict_active, chat_history_active)
                    if msg_active:
                        current_round = (num_step // self.env.num_players()) + 1
                        chat_channel.add_message(player_idx, msg_active, round_idx=current_round)
                        
                    # 2. Peer player speaks
                    peer_idx = 1 - player_idx if len(agent_list) == 2 else 0
                    if peer_idx != player_idx:
                        obs_dict_peer = self.openspiel_observation_to_dict(peer_idx, str(self.env))
                        obs_dict_peer['env_name'] = self.game_name

                        legal_actions_peer = self.env.legal_actions(peer_idx)
                        obs_dict_peer['openspiel_legal_actions'] = legal_actions_peer
                        valid_action_peer = [self.env.action_to_string(a) for a in legal_actions_peer]
                        obs_dict_peer['legal_moves'] = self.openspiel_action_to_agent(valid_action_peer)

                        chat_history_peer = chat_channel.get_recent_window(peer_idx)
                        msg_peer, _ = agent_list[peer_idx].chat_step(obs_dict_peer, chat_history_peer)
                        if msg_peer:
                            current_round = (num_step // self.env.num_players()) + 1
                            chat_channel.add_message(peer_idx, msg_peer, round_idx=current_round)
                
                # init step
                _step = Step(agent_list[player_idx].agent_name)
                _step.set_model_name(model_list[player_idx].nick_name)

                try:
                    observations = self.env.observation_string()
                except Exception as e:
                    observations = str(self.env)

                observation_dict = self.openspiel_observation_to_dict(
                    self.env.current_player(), observations)
                observation_dict['state'] = self.env

                legal_actions = self.env.legal_actions(player_idx)
                observation_dict['openspiel_legal_actions'] = legal_actions
                valid_action = [self.env.action_to_string(
                    a) for a in legal_actions]
                valid_action = self.openspiel_action_to_agent(valid_action)

                observation_dict['legal_moves'] = valid_action
                observation_dict['env_name'] = self.game_name
                observation_dict['player_idx'] = player_idx
                observation_dict['chat_context'] = chat_channel.get_recent_window(player_idx) if all(getattr(a, "enable_chat", False) for a in agent_list) else ""
                
                if len(legal_actions) != 1:
                    action, query_list = agent_list[player_idx].step(
                        observation_dict)
                else:
                    action, query_list = valid_action[0], []

                act = self.quick_action_memory_for_llm.get(
                    player_idx, [])

                act.append(action)
                self.quick_action_memory_for_llm[player_idx] = act

                observation_dict.pop('state')
                _step.set_observation(copy.deepcopy(observation_dict))
                # _step.set_observation(observation_dict)
                self.logger.info(
                    f"openspiel_game_legal_action:{legal_actions}")

                self.logger.info(f"validMove:{valid_action}")

                for q in query_list:
                    _step.add_query(q)

                self.logger.info(
                    f"player: {player_idx} agent:{agent_list[player_idx].agent_name}, action: {action}")

                _step.set_move(action)
                _match.add_step(_step)
                game_action = self.agent_action_to_openspiel(action)
                self.logger.info(f"game_action:{game_action}")
                num_step += 1

                if not self.is_valid_move(game_action, legal_actions):
                    game_action = None
                    agent_name = agent_list[player_idx].agent_name
                    self.logger.info(f"agent {agent_name} made a invalid step")
                    _match.agents_at_fault.append(agent_name)
                    _match.status = "Abnormal"
                    self.status = "Abnormal"
                    break

                self.env.apply_action(game_action)
                # inform other opponents
                for idx, agent in enumerate(agent_list):
                    if player_idx != idx:
                        agent.inform_action(self.env, player_idx, game_action)

        results = self.env.returns()
        if results[0] > results[1]:
            # player 0 wins
            winner_name = agent_list[0].agent_name + \
                "_"+agent_list[0].model.nick_name
            _match.loser_score = results[1]
            _match.winner_score = results[0]
        elif results[1] > results[0]:
            # player 1 wins
            winner_name = agent_list[1].agent_name + \
                "_"+agent_list[1].model.nick_name
            _match.loser_score = results[0]
            _match.winner_score = results[1]
        else:
            # draw
            winner_name = ""

        _match.set_winner(winner_name)
        tracker.add_match(_match)
        if _match.winner != "":
            self.logger.info(f"The winner is {_match.winner}")
        else:
            self.logger.info("There are no winner in this game.")
            
        # [LTM Integration] Post-game updates
        q_mem = self.quick_action_memory_for_llm
        max_turns = max(len(q_mem.get(0, [])), len(q_mem.get(1, []))) if q_mem else 0
        
        for agent_idx, agent in enumerate(agent_list):
            if hasattr(agent, 'post_game_update'):
                agent_history = ""
                
                if self.game_name == 'breakthrough':
                    you_color = ("Black ('b'), advancing downward from Row 8 toward Row 1"
                                 if agent_idx == 0 else
                                 "White ('w'), advancing upward from Row 1 toward Row 8")
                    opp_color = "White ('w')" if agent_idx == 0 else "Black ('b')"
                    agent_history += (
                        f"[Player Context] You play as {you_color}. "
                        f"The opponent plays as {opp_color}.\n"
                        f"[Position Legend] Each [Position] line shows the board before that player's move. "
                        f"Format: a list of row strings from Row 8 (top) to Row 1 (bottom), "
                        f"columns a-c left to right. "
                        f"'b'=Black piece, 'w'=White piece, '.'=empty square.\n\n"
                    )
                elif self.game_name == 'tictactoe':
                    you_symbol = "X (Crosses)" if agent_idx == 0 else "O (Noughts)"
                    opp_symbol = "O (Noughts)" if agent_idx == 0 else "X (Crosses)"
                    agent_history += (
                        f"[Player Context] You play as {you_symbol}. "
                        f"The opponent plays as {opp_symbol}.\n"
                        f"[Position Legend] Each [Position] line shows the board before that player's move. "
                        f"Format: a list of row strings from Row 1 (top) to Row 3 (bottom), "
                        f"columns C1-C3 left to right. "
                        f"'x'=Cross, 'o'=Nought, '.'=empty.\n\n"
                    )
                elif self.game_name == 'connect4':
                    you_symbol = "X (Red)" if agent_idx == 0 else "O (Yellow)"
                    opp_symbol = "O (Yellow)" if agent_idx == 0 else "X (Red)"
                    agent_history += (
                        f"[Player Context] You play as {you_symbol}. "
                        f"The opponent plays as {opp_symbol}.\n"
                        f"[Position Legend] Each [Position] line shows the board before that player's move. "
                        f"Format: a list of row strings from Row 6 (top) to Row 1 (bottom), "
                        f"columns C1-C7 left to right. "
                        f"'x'=Red, 'o'=Yellow, '.'=empty.\n\n"
                    )
                elif self.game_name == 'python_iterated_prisoners_dilemma':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows the round history of decisions so far. "
                        f"Format: Round X: You=[Silent/Testify], Opponent=[Silent/Testify].\n\n"
                    )
                elif self.game_name == 'kuhn_poker':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows your private card and the betting history. "
                        f"Format: Your card: [Card]. Betting history: [Moves].\n\n"
                    )
                elif self.game_name == 'liars_dice':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows the face value of your private die. "
                        f"Format: Your die: [1-6].\n\n"
                    )
                elif self.game_name == 'nim':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows the match counts for the 4 piles. "
                        f"Format: Pile 1: x, Pile 2: y, Pile 3: z, Pile 4: w.\n\n"
                    )
                elif self.game_name == 'pig':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows your banked score, opponent banked score, and turn total. "
                        f"Format: Your score: x, Opponent score: y, Turn total: z.\n\n"
                    )
                elif self.game_name == 'negotiation':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows the remaining item pool, your private valuations, turn stage, and opponent's last proposal/utterance. "
                        f"Format: Pool: [Peppers, Strawberries, Cherries], Your values: [v1, v2, v3], Stage: [Proposal/Utterance], Opponent Proposal: [p1, p2, p3], Opponent Utterance: [u1, u2, u3].\n\n"
                    )
                elif self.game_name == 'first_sealed_auction':
                    agent_history += (
                        f"[Player Context] You play as Player {agent_idx + 1}. "
                        f"The opponent plays as Player {(1 - agent_idx) + 1}.\n"
                        f"[Position Legend] Each [Position] line shows your private valuation. "
                        f"Format: Your private valuation: x.\n\n"
                    )
                else:
                    you_num = 1 if agent_idx == 0 else 2
                    opp_num = 2 if agent_idx == 0 else 1
                    agent_history += (
                        f"[Player Context] You play as Player {you_num}. "
                        f"The opponent plays as Player {opp_num}.\n"
                        f"[Position Legend] Each [Position] line shows the game state before that player's move.\n\n"
                    )

                chat_enabled = all(getattr(a, "enable_chat", False) for a in agent_list)
                chat_by_round = {}
                if chat_enabled:
                    for msg in chat_channel.transcript:
                        r = msg.get("round", 1)
                        if r not in chat_by_round:
                            chat_by_round[r] = []
                        chat_by_round[r].append(msg)
                
                # Pre-filter steps by player to correctly align with q_mem even in non-alternating games
                p0_steps = [s for s in _match.steps if s.observation.get('player_idx') == 0]
                p1_steps = [s for s in _match.steps if s.observation.get('player_idx') == 1]
                
                for t in range(max_turns):
                    r = t + 1
                    agent_history += f"Round {r}:\n"
                    round_chat = chat_by_round.get(r, [])
                    
                    if len(round_chat) > 0:
                        for msg in round_chat[0:2]:
                            prefix = "You" if msg["speaker"] == agent_idx else "Opponent"
                            agent_history += f"  [Chat] {prefix}: {msg['message']}\n"
                            
                    if t < len(q_mem.get(0, [])):
                        prefix = "You" if agent_idx == 0 else "Opponent"
                        if t < len(p0_steps):
                            board = p0_steps[t].observation.get('board', '')
                            if board:
                                if prefix == "Opponent" and hasattr(self, 'get_opponent_board_state'):
                                    board = self.get_opponent_board_state(board)
                                agent_history += f"  [Position]: {board}\n"
                        agent_history += f"  [Move] {prefix}: {q_mem[0][t]}\n\n"
                    else:
                        agent_history += "\n"
                        
                    if len(round_chat) > 2:
                        for msg in round_chat[2:4]:
                            prefix = "You" if msg["speaker"] == agent_idx else "Opponent"
                            agent_history += f"  [Chat] {prefix}: {msg['message']}\n"
                            
                    if t < len(q_mem.get(1, [])):
                        prefix = "You" if agent_idx == 1 else "Opponent"
                        if t < len(p1_steps):
                            board = p1_steps[t].observation.get('board', '')
                            if board:
                                if prefix == "Opponent" and hasattr(self, 'get_opponent_board_state'):
                                    board = self.get_opponent_board_state(board)
                                agent_history += f"  [Position]: {board}\n"
                        agent_history += f"  [Move] {prefix}: {q_mem[1][t]}\n\n"
                    else:
                        agent_history += "\n"
                        
                your_score = results[agent_idx]
                opp_score = results[1 - agent_idx] if len(results) > 1 else results[0]
                agent_history += f"Game Outcome: Your score={your_score}, Opponent score={opp_score}"
                
                try:
                    agent.post_game_update(agent_history, final_board_state=str(self.env), env_name=self.game_name)
                except TypeError:
                    try:
                        agent.post_game_update(agent_history, final_board_state=str(self.env))
                    except TypeError:
                        # Fallback for agents that don't accept final_board_state
                        agent.post_game_update(agent_history)

    def openspiel_observation_to_dict(self, current_player_idx, openspiel_obs):
        return {}

    def openspiel_action_to_agent(self, action):
        return action

    def agent_action_to_openspiel(self, action):
        return action

    def is_match_normal(self) -> bool:
        return self.status == 'Normal'

    def is_valid_move(self, move, valid_moves):
        return move in valid_moves and move != None
