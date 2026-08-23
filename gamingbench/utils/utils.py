
import os
import logging
import random
import threading
import numpy as np
import yaml
import concurrent
import json

from concurrent.futures import ThreadPoolExecutor
from box import Box



def get_game_config_path(game):
    config_root = './gamingbench/configs/game_configs'
    if game == 'tictactoe':
        return os.path.join(config_root, 'tictactoe.yaml')
    elif game == 'connect4':
        return os.path.join(config_root, 'connect4.yaml')
    elif game == 'backgammon':
        return os.path.join(config_root, 'backgammon.yaml')
    elif game == 'breakthrough':
        return os.path.join(config_root, 'breakthrough.yaml')
    elif game == 'first_sealed_auction':
        return os.path.join(config_root, 'first_sealed_auction.yaml')
    elif game == 'gin_rummy':
        return os.path.join(config_root, 'gin_rummy.yaml')
    elif game == 'liars_dice':
        return os.path.join(config_root, 'liars_dice.yaml')
    elif game == 'negotiation':
        return os.path.join(config_root, 'negotiation.yaml')
    elif game == 'nim':
        return os.path.join(config_root, 'nim.yaml')
    elif game == 'pig':
        return os.path.join(config_root, 'pig.yaml')
    elif game == 'kuhn_poker':
        return os.path.join(config_root, 'kuhn_poker.yaml')
    else:
        raise NotImplementedError


def load_game(game_config_path):
    from gamingbench import games
    game_config = Box.from_yaml(
        filename=game_config_path, Loader=yaml.FullLoader)
    return getattr(games, game_config.game_name)(game_config)


def load_config(config_path):
    config = Box.from_yaml(
        filename=config_path, Loader=yaml.FullLoader)

    return config


def load_agent(agent_config_path, **kwargs):
    from gamingbench import agents
    agent_config = Box.from_yaml(
        filename=agent_config_path, Loader=yaml.FullLoader)
    return getattr(agents, agent_config.agent_name)(agent_config, **kwargs)


def load_model(model_config_path):
    from gamingbench import models
    model_config = Box.from_yaml(
        filename=model_config_path, Loader=yaml.FullLoader)
    return getattr(models, model_config.model_type)(model_config)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_logger(logger_path, debug=False, rm_existed=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    if rm_existed and os.path.exists(logger_path):
        os.remove(logger_path)

    fh = logging.FileHandler(logger_path)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    return logger


def parallel_func(worker, arg_list, num_workers=20):
    results = []
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for idx, arg in enumerate(arg_list):
            futures.append(executor.submit(worker, arg))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def strip_thinking_block(text: str) -> str:
    # Strip Gemini thought blocks
    if '</GEMINI_THOUGHT>' in text:
        text = text.rsplit('</GEMINI_THOUGHT>', 1)[-1]
    elif '<GEMINI_THOUGHT>' in text:
        text = text.split('<GEMINI_THOUGHT>', 1)[0]

    # DeepSeek/Qwen3 sometimes start generation AFTER the <think> tag,
    # meaning the output only contains the closing </think> tag.
    if '</think>' in text:
        # Split on the closing tag and keep everything after the final one
        text = text.rsplit('</think>', 1)[-1]
    elif '<think>' in text:
        # Handle truncated/unclosed <think> blocks
        text = text.split('<think>', 1)[0]
        
    if '</thought>' in text:
        text = text.rsplit('</thought>', 1)[-1]
    elif '<thought>' in text:
        text = text.split('<thought>', 1)[0]
        
    # Strip summary tags globally to prevent them leaking into chat/action channels
    import re
    text = re.sub(r"<summary>.*?</summary>", "", text, flags=re.DOTALL)
    
    return text.strip()


def strip_chat_tags(text: str) -> str:
    import re
    # Try to extract content between <chat>...</chat> or <msg>...</msg> tags first.
    # We use re.DOTALL to capture multiline messages.
    match = re.search(r'<(?:chat|msg|message|output)>(.*?)</(?:chat|msg|message|output)>', text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        # Fallback: if no valid closing tag is found but an opening tag exists, take everything after it.
        match = re.search(r'<(?:chat|msg|message|output)>(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # If no tags are found at all, fallback to old behavior of just stripping known tags if they got malformed
            text = re.sub(r'</?(?:chat|msg|message|thought|output)>', '', text, flags=re.IGNORECASE)
            
    # Remove prefixes like "You:", "Opponent:", "Player 1:", "Player 2:", "Me:"
    text = re.sub(r'^(You|Opponent|Player \d+|Me):\s*', '', text, flags=re.IGNORECASE|re.MULTILINE)
    # Also strip wrapping quotes if they exist (e.g., You: "hello" -> hello)
    text = re.sub(r'^["\']|["\']$', '', text.strip())
    # Strip summary tags globally to prevent them leaking into chat/action channels
    import re
    text = re.sub(r"<summary>.*?</summary>", "", text, flags=re.DOTALL)
    
    return text.strip()


def load_jsonl(path):
    result = []
    with open(path, 'r') as f:
        for l in f.readlines():
            r = json.loads(l)
            result.append(r)
    return result


def save_jsonl(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.writelines(json.dumps(r) + '\n')


class LLMBenchLogger:
    _instance = None

    def __new__(cls, logger_path, debug=False, rm_existed=False):
        if cls._instance is None:
            cls._instance = super(LLMBenchLogger, cls).__new__(cls)
            cls._instance.logger = cls._configure_logger(
                logger_path, debug, rm_existed)
        return cls._instance.logger

    @staticmethod
    def _configure_logger(logger_path, debug, rm_existed):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)

        if rm_existed and os.path.exists(logger_path):
            os.remove(logger_path)

        fh = logging.FileHandler(logger_path)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        return logger


class _ThreadFilter(logging.Filter):
    """Allows a FileHandler to receive records only from the thread that created it.

    Used to route per-game log messages to a game-specific file even when
    multiple games run in parallel threads sharing the same root logger.
    """
    def __init__(self, thread_id: int):
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record):
        return threading.current_thread().ident == self._thread_id


def add_game_log_handler(log_path: str) -> logging.FileHandler:
    """Attach a per-game FileHandler to the shared logger for the calling thread.

    The handler is thread-filtered: only records emitted by the calling thread
    are written to log_path.  Call remove_game_log_handler() when the game ends.

    Args:
        log_path: Absolute path of the per-game .log file to create.

    Returns:
        The FileHandler that was added (needed to remove it later).
    """
    logger = logging.getLogger(__name__)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    fh.addFilter(_ThreadFilter(threading.current_thread().ident))
    logger.addHandler(fh)
    return fh


def remove_game_log_handler(fh: logging.FileHandler) -> None:
    """Remove and close the per-game FileHandler returned by add_game_log_handler()."""
    logger = logging.getLogger(__name__)
    logger.removeHandler(fh)
    fh.close()

def query_with_thinking_validation(model, messages, stop=None, prompt_type='move', max_retries=2):
    """
    Executes a model query and validates that reasoning models emit thinking tags.
    If thinking tags are expected but missing, retries up to max_retries.
    If still missing, returns an empty string to allow graceful fallbacks.
    """
    generations, _, _ = model.query(messages, n=1, stop=stop, prompt_type=prompt_type)
    raw_generation = generations[0]
    
    thinking_enabled = getattr(model, 'enable_thinking', False)
    
    retries = 0
    while thinking_enabled and not any(tag in raw_generation for tag in ["<think>", "</think>", "<thought>", "</thought>"]) and retries < max_retries:
        retries += 1
        if hasattr(model, 'logger') and model.logger:
            model.logger.warning(f"Missing thinking tag in generation, retrying ({retries}/{max_retries})...")
        generations, _, _ = model.query(messages, n=1, stop=stop, prompt_type=prompt_type)
        raw_generation = generations[0]
        
    if thinking_enabled and not any(tag in raw_generation for tag in ["<think>", "</think>", "<thought>", "</thought>"]):
        if hasattr(model, 'logger') and model.logger:
            model.logger.error("Failed to generate thinking tags after retries. Returning empty string.")
        return ""
        
    return raw_generation

def truncate_game_history(history_str: str, max_chars: int = 8000) -> str:
    """Truncate the middle of a game history if it exceeds max_chars to prevent context window limits."""
    if len(history_str) <= max_chars:
        return history_str
    half = max_chars // 2
    return history_str[:half] + "\n\n...[TRUNCATED MIDDLE TO SAVE CONTEXT]...\n\n" + history_str[-half:]
