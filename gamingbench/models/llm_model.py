from gamingbench.models.base_model import BaseModel


class LLMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def query(self, messages, n, stop, prompt_type, **kwargs):
        from gamingbench.chat.chat import chat_llm
        assert prompt_type in ['move', 'plan', 'vote', 'critique']
        
        seed = kwargs.get('seed', None)
        effective_seed = seed if seed is not None else getattr(self, 'llm_seed', None)
        
        responses = chat_llm(
            messages=messages,
            model=self.model_path,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=n,
            timeout=self.timeout,
            stop=stop,
            thinking_budget=self.thinking_budget,
            chat_seed=effective_seed
        )
        generations = responses['generations']
        completion_tokens = responses['completion_tokens']
        prompt_tokens = responses['prompt_tokens']
        return generations, completion_tokens, prompt_tokens
