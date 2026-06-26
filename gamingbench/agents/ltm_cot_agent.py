from gamingbench.agents.ltm_agent import LTMAgent
from gamingbench.prompts.step_prompts.cot_agent import construct_step_prompt

class LTMCotAgent(LTMAgent):
    def __init__(self, config, **kwargs):
        super(LTMCotAgent, self).__init__(config, **kwargs)
        
        # Override the step prompt constructor to use Chain of Thought
        self.step_prompt_constructor = construct_step_prompt

    def step(self, observations):
        """
        Runs the full LTM-integrated step, including optional summarization.
        Overrides LTMAgent.step to explicitly pass enable_thinking=False
        for fast CoT gameplay moves.
        """
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []
        
        # 1. Summarization check (before action)
        if self.move_count > 0 and self.move_count % self.summarize_every == 0:
            sum_query = self._run_window_summarization(observations)
            if sum_query:
                query_list.append(sum_query)

        # 2. Prepare action prompt
        system_prompt, observation_prompt = self._build_prompts(observations)

        # Borrow the action regex from the base step prompt constructor.
        step_instruct = self.step_prompt_constructor(observations)
        regex = step_instruct['regex']
        action_format = step_instruct.get('format', observations.get('legal_moves', []))

        # Build action reminder (same logic as base CoT step prompt).
        legal_moves = observations.get('legal_moves', [])
        if len(legal_moves) <= 10:
            action_reminder = f"Remember, you can only choose one move from the legal actions which is {legal_moves}"
        else:
            action_reminder = "Remember, you can only choose one move from the legal actions."

        # Check which databases are loaded for this turn.
        has_opponent_ltm = bool(
            self.current_opponent_key and self.ltm_store.get(self.current_opponent_key)
        )
        has_self_ltm = bool(self.self_ltm_store.get("__self__"))

        if has_opponent_ltm or has_self_ltm:
            # 3-stage structured Thought format — signal evaluation is woven into
            # the reasoning itself so the agent cannot skip it or apply it post-hoc.
            from gamingbench.prompts.regex_and_format import get_step_env_regex_and_format
            _, fmt = get_step_env_regex_and_format(observations.get('env_name', ''))

            scan_sources = []
            if has_self_ltm:
                scan_sources.append("SELF-REPUTATION DATABASE")
            if has_opponent_ltm:
                scan_sources.append("OPPONENT REPUTATION DATABASE")
            scan_label = " and ".join(scan_sources)

            step_prompt = f"""Reason through your move using the three stages below. All three stages are part of your thinking and must appear in your output.

Your output must be in the following format strictly:

Thought:
[Signal Scan] For each signal in your {scan_label}, carefully reason through whether its 'When' condition is met by the current board state and game context. Conclude clearly whether it fires.

[Policy Synthesis] Synthesize the active policies from all firing signals into a coherent strategic directive for this move.

[Move Reasoning] Given the board state and your synthesized policy, reason about the legal moves and choose the best one.

Action:
Your action wrapped by <>, i.e., {fmt}

{action_reminder}
"""
        else:
            # No databases yet — use the standard CoT step prompt unchanged.
            step_prompt = step_instruct['prompt']

        observation_prompt = observation_prompt + '\n' + step_prompt

        msgs = self.construct_init_messages(
            system_prompt, observation_prompt)

        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""

        # 3. Query LLM for move (explicitly disabling native thinking)
        for attempt in range(max_retries):
            responses, query = self.llm_query(
                msgs, n=self.num_generations, stop=None, prompt_type='move', enable_thinking=False)
            query_list.append(query)

            if attempt == 0:
                self.logger.info(f'Prompt: {observation_prompt}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')

            moves = self.parse_with_regex(responses, regex)
            if len(moves) != 0:
                move = self.post_processing(moves, majority_vote=self.voting)
                # Normalize brackets and asterisks from move and valid_moves list to handle potential mismatch
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
                    break
                else:
                    error_msg = f"Invalid move '{move}'. Your move must be one of the legal actions: {valid_moves}. Please try again."
            else:
                move = ""
                error_msg = f"Failed to extract a valid move format. You must output your action wrapped by <>, i.e., <[a-c][1-8]->[a-c][1-8]>. Legal actions: {valid_moves}. Please try again."
            
            if attempt < max_retries - 1:
                self.logger.warning(error_msg)
                msgs.append({"role": "assistant", "content": responses[0]})
                msgs.append({"role": "user", "content": error_msg})

        self.recent_internal_reasoning.append(f"Move {self.move_count} Reasoning:\n{responses[0]}")

        # Log tracing information
        import threading
        # We need the lock from ltm_agent
        from gamingbench.agents.ltm_agent import _trace_log_lock
        log_file = self.ltm_store_path.replace('.json', '_trace.log')
        with _trace_log_lock:
            with open(log_file, "a") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} MOVE {self.move_count} STEP ===\n")
                if getattr(self, 'game_count', 0) == 1:
                    f.write(f"SYSTEM PROMPT:\n{system_prompt}\n")
                    f.write(f"OBSERVATION PROMPT:\n{observation_prompt}\n")
                f.write(f"RESPONSE:\n{responses}\n")
                f.write("=" * 50 + "\n\n")

        self.move_count += 1

        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        if not self.enable_chat:
            return "", None
            
        from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION
        
        self.logger.info('-' * 20 + f'{self.agent_name} Chat Generation' + '-' * 20)
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = self._build_prompts(observations)
        
        observation_prompt = observation_prompt + '\n\n' + CHAT_INSTRUCTION
        
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        try:
            from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move', enable_thinking=False)
            message = strip_thinking_block(responses[0]).strip()
            message = strip_chat_tags(message)
            self.logger.info(f"Chat Generated: {message}")
            return message, query
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return "", None

