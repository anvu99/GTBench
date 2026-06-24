
import re
from gamingbench.utils.history_tracker import Query
from gamingbench.utils import utils
from gamingbench.prompts.chat_prompts import CHAT_HISTORY_INJECTION




class BaseAgent(object):

    def __init__(self, config, **kwargs):
        self.agent_name = config.agent_name
        self.num_generations = config.num_generations
        self.model = None
        self.voting = config.majority_vote
        self.enable_chat = getattr(config, "enable_chat", False)
        self.logger = utils.LLMBenchLogger(None)

    def step(self, observations):
        pass

    def set_game_deep_copy(self, game):
        self.game_env = game

    def set_model(self, model):
        self.model = model

    def llm_query(self, messages, n, stop, prompt_type):
        if self.model == None:
            raise NotImplementedError
        assert prompt_type in ['move', 'plan', 'vote']
        generations, completion_tokens, prompt_tokens = self.model.query(
            messages, n, stop, prompt_type)
        query = self._prompt_to_query(
            messages, prompt_type, generations, token_size=completion_tokens + prompt_tokens)
        return generations, query

    @staticmethod
    def parse_with_regex(content, regex):
        assert isinstance(content, list)
        from gamingbench.utils.utils import strip_thinking_block
        results = []
        for c in content:
            stripped_c = strip_thinking_block(c)
            matched = re.findall(regex, stripped_c)
            if len(matched) > 0:
                results.append(matched)
        return results

    def post_processing(self, moves, majority_vote=False):
        post_moves = []
        for m in moves:
            post_moves.append(m[-1])
        if len(moves) == 0:
            return None
        if majority_vote:
            move = self.majority_vote(post_moves)
        else:
            move = post_moves[-1]
        return move

    @staticmethod
    def _prompt_to_query(msgs, prompt_type, resp, token_size):
        return Query(msgs, prompt_type, resp, token_size)

    @staticmethod
    def majority_vote(candidates):
        high_freq_move_str = max(candidates, key=lambda x: candidates.count(x))
        return high_freq_move_str

    @staticmethod
    def construct_init_messages(system_prompt, user_prompt):
        msgs = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        return msgs

    def inform_action(self, state, player_idx, action):
        pass

    def chat_step(self, observations, chat_history_str: str):
        pass


