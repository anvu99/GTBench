import asyncio
import threading
from typing import List, Dict
from transformers import AutoTokenizer
from gamingbench.models.base_model import BaseModel


class VLLMEngine:
    """Multiton: one AsyncLLMEngine per model_name, shared across all workers/threads."""
    _instances: Dict[str, 'AsyncLLMEngine'] = {}
    _tokenizers: Dict[str, AutoTokenizer] = {}
    _lock = threading.Lock()
    _loop: asyncio.AbstractEventLoop = None
    _thread: threading.Thread = None

    @classmethod
    def _start_background_loop(cls):
        with cls._lock:
            if cls._loop is None:
                cls._loop = asyncio.new_event_loop()
                cls._thread = threading.Thread(
                    target=cls._run_loop,
                    args=(cls._loop,),
                    daemon=True,
                    name="VLLMEngineLoop"
                )
                cls._thread.start()

    @classmethod
    def _run_loop(cls, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    @classmethod
    def get_instance(cls, model_name: str, **engine_kwargs):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        
        cls._start_background_loop()
        
        def init_engine():
            is_qwen3 = "qwen" in model_name.lower() and "qwen3" in model_name.lower().replace("/", "")
            if is_qwen3:
                try:
                    from vllm.config import ReasoningConfig
                    if "reasoning_config" not in engine_kwargs:
                        engine_kwargs["reasoning_config"] = ReasoningConfig()
                except ImportError:
                    pass
                if "reasoning_parser" not in engine_kwargs:
                    engine_kwargs["reasoning_parser"] = "qwen3"
            elif "deepseek" in model_name.lower():
                try:
                    from vllm.config import ReasoningConfig
                    if "reasoning_config" not in engine_kwargs:
                        engine_kwargs["reasoning_config"] = ReasoningConfig()
                except ImportError:
                    pass
                if "reasoning_parser" not in engine_kwargs:
                    engine_kwargs["reasoning_parser"] = "deepseek_r1"
                
            engine_args = AsyncEngineArgs(model=model_name, **engine_kwargs)
            engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            max_len = engine_kwargs.get("max_model_len", 32768)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                model_max_length=max_len,
                trust_remote_code=True
            )
            tokenizer.model_max_length = max_len
            return engine, tokenizer

        with cls._lock:
            if model_name not in cls._instances:
                # Schedule the initialization in the background loop and wait for it
                future = cls._loop.create_future()
                
                def run_init():
                    try:
                        engine, tokenizer = init_engine()
                        cls._loop.call_soon_threadsafe(future.set_result, (engine, tokenizer))
                    except Exception as e:
                        cls._loop.call_soon_threadsafe(future.set_exception, e)
                        
                cls._loop.call_soon_threadsafe(run_init)
                
                # Wait for the initialization to complete (blocks caller thread)
                import time
                while not future.done():
                    time.sleep(0.1)
                    
                engine, tokenizer = future.result()
                cls._instances[model_name] = engine
                cls._tokenizers[model_name] = tokenizer
            
            return cls._instances[model_name]

    @classmethod
    def get_tokenizer(cls, model_name: str) -> AutoTokenizer:
        return cls._tokenizers[model_name]


class VLLMModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Thread-local storage for event loops to bridge sync ThreadPoolExecutor to async vLLM
        self._local = threading.local()
        
        # Parse configs
        self.max_tokens = getattr(config, 'max_tokens', 4096)
        self.temperature = getattr(config, 'temperature', 0.7)
        self.enable_thinking = getattr(config, 'enable_thinking', True)
        self.thinking_budget = getattr(config, 'thinking_budget', 4096)
        
        tensor_parallel_size = getattr(config, 'tensor_parallel_size', 1)
        max_model_len = getattr(config, 'max_model_len', 32768)
        
        engine_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "disable_custom_all_reduce": True,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
        }
        
        # Initialize engine lazily or fetch singleton
        self.engine = VLLMEngine.get_instance(self.model_path, **engine_kwargs)
        self.tokenizer = VLLMEngine.get_tokenizer(self.model_path)
        
        # Match Qwen3 thinking models
        self.is_qwen3 = "qwen" in self.model_path.lower() and "qwen3" in self.model_path.lower().replace("/", "")
        self.enable_thinking = self.enable_thinking if self.is_qwen3 else False
        
        from vllm import SamplingParams
        
        if self.is_qwen3 and self.enable_thinking:
            self.sampling_params = SamplingParams(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                max_tokens=self.max_tokens,
                thinking_token_budget=self.thinking_budget,
            )
        elif self.is_qwen3:
            # No-thinking mode parameters recommended for Qwen3
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                max_tokens=self.max_tokens,
            )
        else:
            self.sampling_params = SamplingParams(
                temperature=self.temperature, 
                max_tokens=self.max_tokens
            )

    def __deepcopy__(self, memo):
        # VLLMModel is stateless and shares singleton engines/tokenizers.
        # Returning self avoids copying self._local (threading.local) which is not picklable.
        return self

    def _strip_thinking(self, text: str) -> str:
        """
        Strips thinking blocks (e.g. <think>...</think> or similar tags) from the generated text.
        """
        if not text:
            return ""
            
        # Standard closing tags for thinking/thought blocks
        for tag in ["</think>", "</thought>"]:
            idx = text.rfind(tag)
            if idx != -1:
                # Return everything after the last closing tag, stripped of whitespace
                return text[idx + len(tag):].strip()
                
        # If no closing tag is found, but an opening tag is present (e.g., truncated generation),
        # return everything before the opening tag.
        for tag in ["<think>", "<thought>"]:
            idx = text.find(tag)
            if idx != -1:
                return text[:idx].strip()
                
        return text.strip()

    def _get_loop(self):
        """Get or create an event loop for the current thread."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def _format_messages(self, messages: List[Dict[str, str]], enable_thinking: bool = None) -> str:
        """Apply the model's native chat template (HuggingFace-standard)."""
        formatted_msgs = []
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            if m["role"] == "system":
                role = "system"
            formatted_msgs.append({"role": role, "content": m["content"]})

        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        if self.is_qwen3:
            return self.tokenizer.apply_chat_template(
                formatted_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return self.tokenizer.apply_chat_template(
            formatted_msgs, tokenize=False, add_generation_prompt=True
        )

    async def _async_query(self, prompt: str, stop: List[str]):
        """Async inner function that actually talks to vLLM Engine."""
        # Use task id for tracking request
        request_id = str(id(asyncio.current_task())) + "_" + str(id(prompt))
        
        from vllm import SamplingParams
        
        # Handle custom stops
        params = self.sampling_params
        if stop:
            stop_seqs = [stop] if isinstance(stop, str) else stop
            # clone params to add stop
            params = SamplingParams(
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
                min_p=params.min_p,
                presence_penalty=params.presence_penalty,
                repetition_penalty=params.repetition_penalty,
                max_tokens=params.max_tokens,
                thinking_token_budget=getattr(params, 'thinking_token_budget', None),
                stop=stop_seqs
            )

        results = self.engine.generate(prompt, params, request_id)
        
        final = None
        async for result in results:
            final = result
            
        if final:
            text = final.outputs[0].text
            completion_tokens = len(final.outputs[0].token_ids)
            prompt_tokens = len(final.prompt_token_ids)
            
            return text, completion_tokens, prompt_tokens
            
        return "", 0, 0

    def query(self, messages, n, stop, prompt_type):
        """
        Synchronous interface required by GTBench BaseModel.
        """
        assert prompt_type in ['move', 'plan', 'vote']
        
        prompt = self._format_messages(messages)
        
        loop = VLLMEngine._loop
        
        response_texts = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        
        # Run n queries sequentially scheduled on the background thread's loop
        for _ in range(n):
            future = asyncio.run_coroutine_threadsafe(self._async_query(prompt, stop), loop)
            text, comp_toks, prmpt_toks = future.result() # Blocks caller thread until done
            
            retries = 0
            # Blind retry (Option B): if the stripped output is completely empty
            while not self._strip_thinking(text).strip() and retries < 2:
                retries += 1
                future = asyncio.run_coroutine_threadsafe(self._async_query(prompt, stop), loop)
                new_text, new_comp_toks, new_prmpt_toks = future.result()
                text = new_text
                comp_toks += new_comp_toks
                prmpt_toks += new_prmpt_toks
                
            response_texts.append(text)
            total_completion_tokens += comp_toks
            total_prompt_tokens += prmpt_toks
            
        return response_texts, total_completion_tokens, total_prompt_tokens
