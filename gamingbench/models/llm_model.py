from gamingbench.models.base_model import BaseModel


class LLMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def query(self, messages, n, stop, prompt_type):
        from gamingbench.chat.chat import chat_llm
        assert prompt_type in ['move', 'plan', 'vote']
        responses = chat_llm(
            messages=messages,
            model=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            timeout=self.timeout,
            stop=stop,
            thinking_budget=self.thinking_budget
        )
        generations = responses['generations']
        completion_tokens = responses['completion_tokens']
        prompt_tokens = responses['prompt_tokens']
        return generations, completion_tokens, prompt_tokens
