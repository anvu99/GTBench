import os
import json
from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.sw.agentpro_store import AgentProStore
from gamingbench.prompts.agentpro_prompts import (
    AGENTPRO_INJECTION_PROMPT,
    AGENTPRO_BELIEF_PROMPT,
    AGENTPRO_CHAT_BELIEF_PROMPT,
    AGENTPRO_REFLECTION_PROMPT
)

class AgentProAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(AgentProAgent, self).__init__(config, **kwargs)
        
        self.agentpro_store_path = getattr(config, "agentpro_store_path", "agentpro_store.json")
        self.agentpro_store = AgentProStore()
        
        if os.path.exists(self.agentpro_store_path):
            self.agentpro_store.load(self.agentpro_store_path)
            
        self._current_belief = {"ses": "", "ops": "", "opo": ""}
        self._current_summary = {"rea": "", "ref": ""}
        self._opponent_key = "unknown"
        self._game_intro = ""
        self.current_game_name = "unknown"

    def set_storage_dir(self, storage_dir):
        """Called by main.py to align storage with the run's experiment folder."""
        base = os.path.basename(self.agentpro_store_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
                
        self.agentpro_store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.agentpro_store_path):
            self.agentpro_store.load(self.agentpro_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking."""
        if os.path.exists(self.agentpro_store_path):
            self.agentpro_store.load(self.agentpro_store_path)
            
        # AgentPro does not use list-based opponent_keys for memory (like N-player games might send),
        # so we convert it to a string if it's a list.
        if isinstance(opponent_key, list):
            self._opponent_key = "+".join(opponent_key)
        else:
            self._opponent_key = opponent_key
            
        self._game_intro = game_intro
        
        # Load prior belief and summary for this opponent (Q4 - Faithfulness)
        belief, summary = self.agentpro_store.get(self.current_game_name, self._opponent_key)
        
        # Reset belief for the new game, but keep summary (which holds the cross-game policy)
        self._current_belief = {"ses": "", "ops": "", "opo": ""}
        self._current_summary = summary

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        # Inject the policy (Behavioral Guideline) from previous games if it exists
        ref = self._current_summary.get("ref", "").strip()
        if ref:
            # We don't have self.player_id reliably as 1-indexed, but we can extract it or use agent_name
            pid = getattr(self, 'player_id', '0').replace('p', '')
            player_index = str(int(pid) + 1) if pid.isdigit() else pid
            
            injection = AGENTPRO_INJECTION_PROMPT.format(
                player_index=player_index,
                ref=ref
            )
            # Find the game intro in the observation prompt and insert injection after it (Option A)
            from gamingbench.prompts.observation_prompts import construct_game_intro
            env_name = getattr(self, 'current_game_name', observations.get('env_name', 'unknown'))
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
            
            # Simple prepend if we can't find exact game intro
            if game_intro in observation_prompt:
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + injection, 1)
            else:
                observation_prompt = injection + "\n\n" + observation_prompt
                
        return system_prompt, observation_prompt

    def chat_step(self, observations, chat_history_str: str):
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        self.logger.info('-' * 20 + f'{self.agent_name} Chat Generation' + '-' * 20)
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = self._build_prompts(observations)
        
        pid = getattr(self, 'player_id', '0').replace('p', '')
        player_index = str(int(pid) + 1) if pid.isdigit() else pid
        
        if any(v for v in self._current_belief.values()):
            history_belief = (
                "In a previous round, you have established a belief like this:\n"
                f"your own game situation is: {self._current_belief['ses']}\n"
                f"the game situation of opponents is: {self._current_belief['ops']}\n"
                f"your opponent's opinion of you is: {self._current_belief['opo']}\n"
            )
        else:
            history_belief = ""
            
        chat_prompt = AGENTPRO_CHAT_BELIEF_PROMPT.format(
            history_belief=history_belief,
            player_index=player_index
        )
        
        observation_prompt = observation_prompt + '\n\n' + chat_prompt
        
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        try:
            from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags
            import re
            
            max_retries = 3
            message = ""
            query_list = []
            
            for attempt in range(max_retries):
                responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
                query_list.append(query)
                
                raw_response = strip_thinking_block(responses[0]).strip()
                
                if attempt == 0:
                    self.logger.info(f'Chat Prompt: {msgs[1]["content"]}')
                self.logger.info(f'Chat Raw Response (Attempt {attempt+1}): {responses}')
                
                # Extract belief components
                try:
                    ses_match = re.search(r'<ses>(.*?)</ses>', raw_response, re.DOTALL)
                    ops_match = re.search(r'<ops>(.*?)</ops>', raw_response, re.DOTALL)
                    opo_match = re.search(r'<opo>(.*?)</opo>', raw_response, re.DOTALL)
                    
                    if ses_match: self._current_belief['ses'] = ses_match.group(1).strip()
                    if ops_match: self._current_belief['ops'] = ops_match.group(1).strip()
                    if opo_match: self._current_belief['opo'] = opo_match.group(1).strip()
                except Exception as e:
                    self.logger.error(f"Error parsing chat belief fields: {e}")
                
                # Extract chat message
                message = strip_chat_tags(raw_response)
                
                if message:
                    self.logger.info(f"Chat Generated: {message}")
                    break
                else:
                    error_msg = "Failed to extract a valid chat message. You must output a non-empty message wrapped by <chat>...</chat>. Please try again."
                    if attempt < max_retries - 1:
                        self.logger.warning(error_msg)
                        msgs.append({"role": "assistant", "content": responses[0]})
                        msgs.append({"role": "user", "content": error_msg})
            
            # Save belief after EVERY chat step (continuous update)
            self.agentpro_store.update(self.current_game_name, self._opponent_key, self._current_belief, self._current_summary)
            self.agentpro_store.save(self.agentpro_store_path)
            
            if not message:
                self.logger.info(f"Chat Generated (Empty Fallback): {message}")
            return message, query_list
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return "", None

    def step(self, observations):
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []

        system_prompt, observation_prompt = self._build_prompts(observations)
        
        valid_moves = observations.get('legal_moves', [])
        
        pid = getattr(self, 'player_id', '0').replace('p', '')
        player_index = str(int(pid) + 1) if pid.isdigit() else pid
        
        # Format the history belief string
        if any(v for v in self._current_belief.values()):
            history_belief = (
                "In a previous round, you have established a belief like this:\n"
                f"your own game situation is: {self._current_belief['ses']}\n"
                f"the game situation of opponents is: {self._current_belief['ops']}\n"
                f"your opponent's opinion of you is: {self._current_belief['opo']}\n"
            )
        else:
            history_belief = ""
            
        example_action = valid_moves[0] if valid_moves else "action1"

        belief_prompt = AGENTPRO_BELIEF_PROMPT.format(
            history_belief=history_belief,
            player_index=player_index,
            legal_actions=str(valid_moves),
            example_action=example_action
        )
        
        observation_prompt = observation_prompt + '\n\n' + belief_prompt

        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        max_retries = 3
        move = ""

        for attempt in range(max_retries):
            responses, query = self.llm_query(
                msgs, n=self.num_generations, stop=None, prompt_type='move')
            query_list.append(query)

            if attempt == 0:
                self.logger.info(f'Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')

            from gamingbench.utils.utils import strip_thinking_block
            import re
            
            raw_response = strip_thinking_block(responses[0])
            
            # Extract belief components
            try:
                ses_match = re.search(r'<ses>(.*?)</ses>', raw_response, re.DOTALL)
                ops_match = re.search(r'<ops>(.*?)</ops>', raw_response, re.DOTALL)
                opo_match = re.search(r'<opo>(.*?)</opo>', raw_response, re.DOTALL)
                
                if ses_match: self._current_belief['ses'] = ses_match.group(1).strip()
                if ops_match: self._current_belief['ops'] = ops_match.group(1).strip()
                if opo_match: self._current_belief['opo'] = opo_match.group(1).strip()
            except Exception as e:
                self.logger.error(f"Error parsing belief fields: {e}")

            # Extract action (the action is parsed directly from the belief LLM call)
            try:
                # Handle single or double quotes for json format
                action_match = re.search(r'{\s*[\'"]action[\'"]\s*:\s*[\'"](.*?)[\'"]\s*}', raw_response)
                if action_match:
                    move = action_match.group(1).strip()
                else:
                    move = ""
            except Exception as e:
                self.logger.error(f"Error parsing action: {e}")
                move = ""

            if move:
                def clean_action(act):
                    return act.replace('<', '').replace('>', '').replace('*', '').strip()
                cleaned_move = clean_action(move)
                matched_valid_move = None
                for m in valid_moves:
                    if clean_action(m) == cleaned_move:
                        matched_valid_move = m
                        break
                
                if not valid_moves or matched_valid_move is not None:
                    if matched_valid_move is not None:
                        move = matched_valid_move
                    
                    # VALID MOVE FOUND - Break out of retry loop
                    break
                else:
                    error_msg = f"Invalid move '{move}'. Your action must be selected from {valid_moves}. Please try again."
            else:
                error_msg = f"Failed to extract a valid action format. You must output your action as {{\"action\": \"...\"}}. Legal actions: {valid_moves}. Please try again."

            if attempt < max_retries - 1:
                self.logger.warning(error_msg)
                msgs.append({"role": "assistant", "content": responses[0]})
                msgs.append({"role": "user", "content": error_msg})

        # Save belief after EVERY step (faithfulness to believe.py run loop)
        self.agentpro_store.update(self.current_game_name, self._opponent_key, self._current_belief, self._current_summary)
        self.agentpro_store.save(self.agentpro_store_path)

        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        self.current_game_name = env_name.lower()
        self._do_reflection(game_history)

    def _do_reflection(self, game_history: str):
        self.logger.info('-' * 20 + f'{self.agent_name} Post-Game Reflection' + '-' * 20)
        
        pid = getattr(self, 'player_id', '0').replace('p', '')
        player_index = str(int(pid) + 1) if pid.isdigit() else pid
        
        reflection_prompt = AGENTPRO_REFLECTION_PROMPT.format(
            game_name=self.current_game_name,
            game_intro=self._game_intro,
            game_history=game_history,
            ses=self._current_belief.get('ses', ''),
            ops=self._current_belief.get('ops', ''),
            opo=self._current_belief.get('opo', ''),
            player_index=player_index
        )
        
        messages = [{"role": "user", "content": reflection_prompt}]
        
        try:
            from gamingbench.utils.utils import strip_thinking_block
            import re
            
            generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            raw_response = strip_thinking_block(generations[0])
            
            rea_match = re.search(r'<rea>(.*?)</rea>', raw_response, re.DOTALL)
            ref_match = re.search(r'<ref>(.*?)</ref>', raw_response, re.DOTALL)
            
            if rea_match: self._current_summary['rea'] = rea_match.group(1).strip()
            if ref_match: self._current_summary['ref'] = ref_match.group(1).strip()
            
            self.logger.info(f"AgentPro Reflection Complete.\nReason: {self._current_summary.get('rea', '')}\nStrategy: {self._current_summary.get('ref', '')}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate AgentPro reflection: {e}")
            
        # Reset belief for the next game, keep the new summary
        self._current_belief = {"ses": "", "ops": "", "opo": ""}
        
        # Save to store
        self.agentpro_store.update(self.current_game_name, self._opponent_key, self._current_belief, self._current_summary)
        self.agentpro_store.save(self.agentpro_store_path)
