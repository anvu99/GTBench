import re
from gamingbench.utils.history_tracker import Query


class BaseModel(object):

    def __init__(self, config):
        self.model_path = config.llm_model_path
        self.max_tokens = config.max_tokens
        self.timeout = getattr(config, 'timeout', 120)
        self.temperature = getattr(config, 'temperature', 0.7)
        self.nick_name = config.nick_name
        self.thinking_budget = getattr(config, 'thinking_budget', 0)

    def query(self, messages, n, stop, prompt_type, **kwargs):
        pass
