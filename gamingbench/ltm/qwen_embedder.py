import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np


def _last_token_pool(last_hidden_states, attention_mask):
    """Official Qwen3-Embedding pooling function (verbatim from Qwen docs).
    Handles both left-padding (fast path) and right-padding batches correctly.
    """
    # Qwen uses the very last hidden state token to represent the entire sequence embedding.
    # Left padding is the standard format for decoder-only models which allows us to simply take the last token.
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


class QwenEmbedder:
    """Wrapper for Qwen3-Embedding models using raw transformers.
    
    This avoids vLLM which would conflict with the 35B LLM's NCCL group,
    and supports all official best practices (FA2, last-token pooling,
    and query instruction prefixes).
    """
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", gpu_id=0, max_length=8192,
                 instruction="Given a game board state, retrieve the most relevant past game signals or strategies",
                 use_flash_attn=True):
        self.max_length = max_length
        self.instruction = instruction
        device_str = "cpu" if str(gpu_id).lower() == "cpu" else f"cuda:{gpu_id}"

        # padding_side='left' is REQUIRED for last_token_pool to take the fast
        # [:, -1] path, which is what the model was trained to expect.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

        # Flash attention is only supported on GPU
        attn_impl = "flash_attention_2" if use_flash_attn and device_str != "cpu" else "eager"
        
        # CPU doesn't support float16 properly for many ops, fallback to float32 for CPU
        dtype = torch.float32 if device_str == "cpu" else torch.float16
        
        self.model = AutoModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device_str)
        
        self.model.eval()

    def encode(self, texts, is_query=False):
        """Encode one or more texts into L2-normalized embedding vectors.

        Args:
            texts: str or list[str]
            is_query: if True, prepend the instruction prefix (query side).
                      if False, encode raw text (document/centroid side).
        Returns:
            np.ndarray of shape (N, dim) for a list, or (dim,) for a single string.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        if is_query:
            # No space after 'Query:' matches the exact training format
            texts = [f'Instruct: {self.instruction}\nQuery:{t}' for t in texts]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move tokenized batch to model device (tokenizer runs on CPU, model runs on GPU)
        batch = batch.to(self.model.device)

        # Forward pass through the Transformer
        with torch.no_grad():
            outputs = self.model(**batch)

        embeddings = _last_token_pool(outputs.last_hidden_state, batch['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        result = embeddings.cpu().numpy().astype(np.float32)
        return result[0] if single else result
