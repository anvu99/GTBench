"""
SimpleTendencyAgent — the simplest proactive-query-style agent.

Design:
  - Maintains one plain-text "tendency block" per opponent in a JSON file.
  - In-game: injects the block (read-only) above the board state every step.
  - Post-game (per game): 1 LLM call produces a "delta" — a bullet list of
    behaviours observed in this single game with occurrence counts.
  - Post-batch (flush): 1 LLM call merges all per-game deltas into the
    canonical tendency block and writes it to disk.

No embedder, no stat pool, no question generation loop.
"""

import json
import os
import re
import threading

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.tendency_prompts import (
    TENDENCY_REFLECTION_PROMPT,
    TENDENCY_MERGE_PROMPT,
    TENDENCY_INJECTION_BLOCK,
    TENDENCY_EMPTY_BLOCK,
)
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.utils.utils import strip_thinking_block


class SimpleTendencyAgent(PromptAgent):
    """
    Simplest proactive-query agent: tracks opponent behaviour tendencies as a
    plain-text reputation block, updated after each batch of games via two
    lightweight LLM calls.
    """

    # Class-level lock for thread-safe file logging
    _log_lock = threading.Lock()

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.tendency_store_path = getattr(config, "tendency_store_path", "tendency_store.json")
        self.max_behaviours = getattr(config, "max_behaviours", 15)
        self.batch_mode = getattr(config, "batch_mode", True)

        # tendency_store: {opponent_key: {"games_observed": int, "block": str}}
        self.tendency_store: dict = {}
        if os.path.exists(self.tendency_store_path):
            self._load_store()

        # Per-game state
        self.current_opponent_key: str | None = None
        self.current_game_intro: str = ""
        self._last_batch_result: dict | None = None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_store(self):
        try:
            with open(self.tendency_store_path, "r", encoding="utf-8") as f:
                self.tendency_store = json.load(f)
        except (json.JSONDecodeError, OSError):
            self.tendency_store = {}

    def _save_store(self):
        if self.tendency_store_path == "/dev/null":
            return
        with open(self.tendency_store_path, "w", encoding="utf-8") as f:
            json.dump(self.tendency_store, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def set_storage_dir(self, storage_dir: str):
        """Called by main.py to align storage path with the run's experiment folder."""
        self.storage_dir = storage_dir
        base = os.path.basename(self.tendency_store_path)
        self.tendency_store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.tendency_store_path):
            self._load_store()

    def reset_game_state(self, opponent_key, game_intro: str):
        """Called at the start of each game."""
        if not self.batch_mode:
            if os.path.exists(self.tendency_store_path):
                self._load_store()

        self.current_game_intro = game_intro

        # Handle N-player combined mode: join sorted keys
        if isinstance(opponent_key, list):
            self.current_opponent_key = "+".join(sorted(opponent_key))
        else:
            self.current_opponent_key = opponent_key

    # ------------------------------------------------------------------
    # Tendency block access
    # ------------------------------------------------------------------

    def _get_tendency_block(self) -> str:
        """Return the current canonical tendency block for the active opponent."""
        if not self.current_opponent_key:
            return TENDENCY_EMPTY_BLOCK
        entry = self.tendency_store.get(self.current_opponent_key)
        if not entry or not entry.get("block"):
            return TENDENCY_EMPTY_BLOCK
        return entry["block"]

    def _get_games_observed(self) -> int:
        if not self.current_opponent_key:
            return 0
        return self.tendency_store.get(self.current_opponent_key, {}).get("games_observed", 0)

    # ------------------------------------------------------------------
    # In-game: prompt injection
    # ------------------------------------------------------------------

    def _build_prompts(self, observations):
        """Override: inject the tendency block above the board state."""
        system_prompt, observation_prompt = super()._build_prompts(observations)

        env_name = observations["env_name"]
        board_state = construct_observation_prompt(observations, env_name)

        tendency_block = self._get_tendency_block()
        injection = TENDENCY_INJECTION_BLOCK.format(tendency_block=tendency_block)

        # Insert immediately before the board state
        observation_prompt = observation_prompt.replace(
            board_state,
            injection + "\n\n" + board_state,
            1,
        )
        return system_prompt, observation_prompt

    # step() and chat_step() are inherited from PromptAgent unchanged —
    # the only difference is that _build_prompts now injects the tendency block.

    # ------------------------------------------------------------------
    # Post-game: per-game reflection call
    # ------------------------------------------------------------------

    def _log_prompt(self, phase_title: str, prompt: str, raw_answer: str):
        """Log prompt + response to both the run logger and a dedicated file."""
        log_str = (
            f"=== {phase_title} ===\n"
            f"PROMPT:\n{prompt}\n"
            f"RAW ANSWER:\n{raw_answer}\n"
            f"{'=' * (8 + len(phase_title))}\n\n"
        )
        if hasattr(self, "logger"):
            self.logger.info(f"=== {phase_title} ===")
            self.logger.info(f"PROMPT:\n{prompt}")
            self.logger.info(f"RAW ANSWER:\n{raw_answer}")
            self.logger.info("=" * (8 + len(phase_title)))

        with self._log_lock:
            try:
                log_dir = getattr(self, "storage_dir", os.path.dirname(self.tendency_store_path))
                if log_dir and log_dir != "/dev" and log_dir != "/dev/null":
                    log_file = os.path.join(log_dir, f"{self.agent_name}_tendency_processing.log")
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(log_str)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.error(f"Failed to write to tendency processing log: {e}")

    def _run_reflection(self, game_history: str) -> str:
        """
        Single LLM call: given current tendency block + game trajectory,
        produce a delta bullet list of behaviours observed in this game.
        Returns the raw delta text (bullet list).
        """
        current_block = self._get_tendency_block()

        # Personalise the history from the agent's perspective
        if self.agent_name:
            game_history = game_history.replace(self.agent_name, f"{self.agent_name} (You)")

        game_rules = getattr(self, "current_game_intro", "") or ""
        prompt = TENDENCY_REFLECTION_PROMPT.format(
            game_rules=game_rules,
            current_block=current_block,
            game_history=game_history,
        )
        messages = [{"role": "user", "content": prompt}]
        responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type="move")
        raw = responses[0]
        stripped = strip_thinking_block(raw)

        self._log_prompt("PER-GAME TENDENCY REFLECTION", prompt, raw)
        return stripped.strip()

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = "unknown"):
        """
        Called by the framework after each game ends.

        In batch mode: runs the per-game reflection call and stashes the
        delta in _last_batch_result for flush_batch_updates() to consume.
        """
        if not self.current_opponent_key:
            return

        self.logger.info("-" * 20 + f" {self.agent_name} Post-Game Tendency Reflection " + "-" * 20)

        delta_text = self._run_reflection(game_history)

        self.logger.info(f"Delta from this game:\n{delta_text}")

        if self.batch_mode:
            self._last_batch_result = {
                "opponent_key": self.current_opponent_key,
                "delta": delta_text,
            }
            return

        # Non-batch fallback: apply delta immediately (single-game run)
        self._apply_deltas(self.current_opponent_key, [delta_text], num_games=1)
        self._save_store()

    # ------------------------------------------------------------------
    # Post-batch: merge all deltas into the canonical block
    # ------------------------------------------------------------------

    def _apply_deltas(self, opponent_key: str, deltas: list[str], num_games: int):
        """
        Single LLM call that merges all per-game delta bullet lists into the
        existing canonical tendency block and writes the result back to the store.
        """
        current_block = self.tendency_store.get(opponent_key, {}).get("block") or TENDENCY_EMPTY_BLOCK
        games_observed = self.tendency_store.get(opponent_key, {}).get("games_observed", 0)

        # Format the per-game deltas into a numbered multi-section string
        delta_sections = []
        for i, d in enumerate(deltas, start=1):
            body = d.strip() if d.strip() else "(No noteworthy behaviours observed in this game.)"
            delta_sections.append(f"[Game {i}]\n{body}")
        all_deltas_str = "\n\n".join(delta_sections)

        game_rules = getattr(self, "current_game_intro", "") or ""
        prompt = TENDENCY_MERGE_PROMPT.format(
            game_rules=game_rules,
            current_block=current_block,
            all_deltas=all_deltas_str,
            num_games=num_games,
            max_behaviours=self.max_behaviours,
        )
        messages = [{"role": "user", "content": prompt}]
        responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type="move")
        raw = responses[0]
        stripped = strip_thinking_block(raw)

        self._log_prompt("BATCH TENDENCY MERGE", prompt, raw)

        # Extract the updated block from the LLM output
        new_block = self._extract_tendency_block(stripped)
        if not new_block:
            # Fallback: keep the old block rather than corrupt the store
            self.logger.warning(
                "SimpleTendencyAgent: failed to extract a well-formed tendency block from merge response; "
                "keeping existing block."
            )
            new_block = current_block

        self.tendency_store[opponent_key] = {
            "games_observed": games_observed + num_games,
            "block": new_block,
        }

    @staticmethod
    def _extract_tendency_block(text: str) -> str:
        """
        Pull out the === OPPONENT TENDENCY PROFILE === ... ========= block
        from the LLM response. Returns empty string if not found.
        """
        match = re.search(
            r"(=== OPPONENT TENDENCY PROFILE ===.*?=================================)",
            text,
            re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def flush_batch_updates(self, gradient_data: list) -> None:
        """
        Called by main.py after all games in a batch complete.

        gradient_data: list of _last_batch_result dicts, one per game.
          Each dict: {"opponent_key": str, "delta": str}

        Groups deltas by opponent_key, then runs one merge LLM call per
        opponent key and writes the updated store to disk.
        """
        if not gradient_data:
            return

        # Group deltas by opponent key
        grouped: dict[str, list[str]] = {}
        for item in gradient_data:
            if not isinstance(item, dict):
                continue
            key = item.get("opponent_key")
            delta = item.get("delta", "")
            if key:
                grouped.setdefault(key, []).append(delta)

        for opponent_key, deltas in grouped.items():
            self.logger.info(
                f"SimpleTendencyAgent flush: merging {len(deltas)} game delta(s) for opponent '{opponent_key}'"
            )
            self._apply_deltas(opponent_key, deltas, num_games=len(deltas))

        self._save_store()
