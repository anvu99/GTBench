import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from gamingbench.models.base_model import BaseModel


class VLLMServerModel(BaseModel):
    """
    Model client that talks to a running vLLM OpenAI-compatible server.

    Start the server once with start_vllm_server.sh, then submit as many
    experiments as you want — they all share the same loaded model and vLLM
    handles GPU batching automatically.

    Drop-in replacement for VLLMModel: same query() interface, same output
    format, same thinking-tag stripping.
    """

    def __init__(self, config):
        super().__init__(config)

        self.base_url = getattr(config, 'base_url', 'http://localhost:8000/v1')
        self.max_tokens = getattr(config, 'max_tokens', 4096)
        self.temperature = getattr(config, 'temperature', 0.7)
        self.enable_thinking = getattr(config, 'enable_thinking', True)
        self.thinking_budget = getattr(config, 'thinking_budget', 4096)
        self.timeout = getattr(config, 'timeout', 600)
        # Sampling seed for reproducibility. When set, vLLM seeds its RNG
        # per-request so the same prompt + seed yields the same output.
        self.llm_seed = getattr(config, 'llm_seed', None)

        # Match Qwen3 thinking models
        self.is_qwen3 = (
            "qwen" in self.model_path.lower()
            and "qwen3" in self.model_path.lower().replace("/", "")
        )

        # Match Gemma-4 thinking models (google/gemma-4-*)
        # Gemma-4 uses <thinking>...</thinking> tags; vLLM server must be
        # started with --reasoning-parser gemma4.
        self.is_gemma4 = (
            "gemma" in self.model_path.lower()
            and "gemma-4" in self.model_path.lower().replace("_", "-")
        )

        self.enable_thinking = self.enable_thinking if (self.is_qwen3 or self.is_gemma4) else False

        # Lazy-init the openai client (thread-safe: client is stateless)
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        """Return (or create) the OpenAI client pointed at our vLLM server."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from openai import OpenAI
                    except ImportError:
                        raise ImportError(
                            "openai package is required for VLLMServerModel. "
                            "Install with: pip install openai"
                        )
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key="EMPTY",  # vLLM server does not require a real key
                        timeout=self.timeout,
                    )
        return self._client

    # ── Thinking-tag stripping (identical to VLLMModel) ──────────────────────

    def _strip_thinking(self, text: str) -> str:
        """Strip thinking blocks from output.

        Handles both:
          - Qwen3-style:  <think>...</think> or <thought>...</thought>
          - Gemma-4-style: <thinking>...</thinking>
        """
        if not text:
            return ""

        # Closing tags — longest/most-specific first to avoid partial matches
        for tag in ["</thinking>", "</think>", "</thought>"]:
            idx = text.rfind(tag)
            if idx != -1:
                return text[idx + len(tag):].strip()

        # Opening tag only (truncated generation fallback)
        for tag in ["<thinking>", "<think>", "<thought>"]:
            idx = text.find(tag)
            if idx != -1:
                return text[:idx].strip()

        return text.strip()

    # ── Single request ────────────────────────────────────────────────────────

    def _single_query(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]],
        enable_thinking: Optional[bool] = None,
        seed: Optional[int] = None,
    ):
        """
        Fire one chat-completion request to the vLLM server and return
        (text, completion_tokens, prompt_tokens).
        """
        client = self._get_client()

        if enable_thinking is None:
            enable_thinking = self.enable_thinking

        # Build extra_body for thinking control
        extra_body: Dict = {}
        if self.is_qwen3:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
            if enable_thinking and self.thinking_budget:
                extra_body["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}
        elif self.is_gemma4:
            # Gemma-4 has no token budget param — just the template kwarg.
            # The server-side --reasoning-parser gemma4 extracts <thinking> blocks
            # into choice.message.reasoning automatically.
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        # Temperature: use Qwen3-recommended values
        if self.is_qwen3 and enable_thinking:
            temperature = 1.0
            top_p = 0.95
        elif self.is_qwen3:
            temperature = 0.7
            top_p = 0.8
        else:
            temperature = self.temperature
            top_p = 1.0

        # Build stop list
        stop_seqs = None
        if stop:
            stop_seqs = [stop] if isinstance(stop, str) else list(stop)

        # Resolve the seed: per-call override > instance default > None
        effective_seed = seed if seed is not None else self.llm_seed
        if effective_seed is not None:
            extra_body["seed"] = effective_seed

        kwargs = dict(
            model=self.model_path,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_seqs,
            extra_body=extra_body if extra_body else None,
        )

        import time
        from openai import APITimeoutError, APIConnectionError, InternalServerError

        max_retries = 3
        base_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                if attempt == max_retries - 1:
                    raise
                print(f"[VLLMServerModel] API request failed with {type(e).__name__}: {e}. Retrying in {base_delay * (2**attempt)} seconds...")
                time.sleep(base_delay * (2**attempt))

        choice = response.choices[0]
        text = choice.message.content or ""
        
        # vLLM's reasoning parser puts the thinking block into msg.reasoning.
        # Re-wrap it with the model's native tag so _strip_thinking() works
        # correctly and logs show clearly which model produced the thought.
        reasoning = getattr(choice.message, "reasoning", "") or ""
        if reasoning:
            if self.is_gemma4:
                text = f"<thinking>\n{reasoning}\n</thinking>\n{text}"
            else:
                text = f"<think>\n{reasoning}\n</think>\n{text}"

        completion_tokens = response.usage.completion_tokens if response.usage else 0
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0

        return text, completion_tokens, prompt_tokens

    # ── Public interface (matches VLLMModel.query) ────────────────────────────

    def query(
        self,
        messages: List[Dict[str, str]],
        n: int,
        stop,
        prompt_type: str,
        **kwargs,
    ):
        """
        Synchronous interface required by GTBench BaseModel.

        Fires n requests concurrently via a ThreadPoolExecutor so the vLLM
        server sees them all at once and batches them in a single continuous-
        batching window — identical behaviour to VLLMModel.query().

        Returns:
            (response_texts, total_completion_tokens, total_prompt_tokens)
        """
        assert prompt_type in ['move', 'plan', 'vote', 'critique']

        enable_thinking = kwargs.get('enable_thinking', None)
        seed = kwargs.get('seed', None)

        total_completion_tokens = 0
        total_prompt_tokens = 0
        response_texts = []

        # Fire all n requests in parallel
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(self._single_query, messages, stop, enable_thinking, seed)
                for _ in range(n)
            ]
            raw_results = [f.result() for f in futures]

        for text, comp_toks, prmpt_toks in raw_results:
            total_completion_tokens += comp_toks
            total_prompt_tokens += prmpt_toks

            # Blind retry if stripped output is completely empty (up to 2x)
            retries = 0
            while not self._strip_thinking(text).strip() and retries < 2:
                retries += 1
                text, new_comp, new_prmpt = self._single_query(
                    messages, stop, enable_thinking, seed
                )
                total_completion_tokens += new_comp
                total_prompt_tokens += new_prmpt

            response_texts.append(text)

        return response_texts, total_completion_tokens, total_prompt_tokens

    def __deepcopy__(self, memo):
        # VLLMServerModel is stateless (HTTP client is thread-safe).
        # Returning self avoids copying the lock and client unnecessarily.
        return self
