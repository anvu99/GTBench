from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.prompts.step_prompts.cot_agent import construct_step_prompt


class CoTAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(CoTAgent, self).__init__(config, **kwargs)

        self.step_prompt_constructor = construct_step_prompt

    def step(self, observations):
        """
        Runs the step with a validation loop and retries, disabling native thinking.
        """
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []

        # Prepare action prompt
        system_prompt, observation_prompt = self._build_prompts(observations)

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead. Anticipate the opponent's likely responses to your move, and consider long-term strategic implications rather than just immediate tactical gains."

        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']

        msgs = self.construct_init_messages(
            system_prompt, observation_prompt)

        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""

        # Query LLM for move (explicitly disabling native thinking)
        for attempt in range(max_retries):
            responses, query = self.llm_query(
                msgs, n=self.num_generations, stop=None, prompt_type='move', enable_thinking=False)
            query_list.append(query)

            if attempt == 0:
                self.logger.info(f'Prompt: {observation_prompt}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')

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

        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list
