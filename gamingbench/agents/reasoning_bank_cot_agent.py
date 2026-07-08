from gamingbench.agents.reasoning_bank_agent import ReasoningBankAgent
from gamingbench.prompts.step_prompts.cot_agent import construct_step_prompt

class ReasoningBankCotAgent(ReasoningBankAgent):
    def __init__(self, config, **kwargs):
        super(ReasoningBankCotAgent, self).__init__(config, **kwargs)
        # Override the step prompt constructor to use Chain of Thought
        self.step_prompt_constructor = construct_step_prompt

    def step(self, observations):
        """
        Runs the full ReasoningBank-integrated step.
        Overrides ReasoningBankAgent.step to explicitly pass enable_thinking=False
        for fast CoT gameplay moves.
        """
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []

        system_prompt, observation_prompt = self._build_prompts(observations)

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead. Anticipate the opponent's likely responses to your move, and consider long-term strategic implications rather than just immediate tactical gains."

        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']

        msgs = self.construct_init_messages(system_prompt, observation_prompt)

        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""
        final_responses = None

        # Query LLM for move (explicitly disabling native thinking)
        for attempt in range(max_retries):
            responses, query = self.llm_query(
                msgs, n=self.num_generations, stop=None, prompt_type='move', enable_thinking=False)
            query_list.append(query)

            if attempt == 0:
                self.logger.info(f'Prompt: {observation_prompt}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')
            final_responses = responses

            moves = self.parse_with_regex(responses, regex)
            if len(moves) != 0:
                move = self.post_processing(moves, majority_vote=getattr(self, "voting", False))
                
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

        # Save trajectory for post-game memory extraction
        if final_responses:
            self.current_trajectory.append(f"Prompt:\n{observation_prompt}\n\nResponse:\n{final_responses[0]}")

        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        if not getattr(self, "enable_chat", False):
            return "", None
            
        from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION
        
        self.logger.info('-' * 20 + f'{self.agent_name} Chat Generation' + '-' * 20)
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = self._build_prompts(observations)
        observation_prompt = observation_prompt + '\n\n' + CHAT_INSTRUCTION
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        try:
            from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags
            # Disable native thinking for chat generation
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move', enable_thinking=False)
            message = strip_thinking_block(responses[0]).strip()
            message = strip_chat_tags(message)
            self.logger.info(f"Chat Generated: {message}")
            return message, query
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return "", None
