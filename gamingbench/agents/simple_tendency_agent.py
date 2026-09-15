"""
SimpleTendencyAgent v2 — Structured Tendency Memory.

Design:
  - Maintains structured CONDITIONAL + INFERENCE memory per opponent in a JSON file.
  - CONDITIONAL: "When game is in state X, opponent tends to take action Y."
    Fields: game_state (neutral trigger), actions (list of {id, description, count})
  - INFERENCE: "When opponent does public action P, it reveals hidden information H."
    Fields: public_action (observable trigger), hidden_information (list of {id, description, count})
  - In-game: injects the rendered profile (read-only) above the board state every step.
  - Post-game: 1 LLM call produces a JSON delta list.
    On JSON parse failure, retries once. If both fail, skips this game's delta.
  - Post-batch: 1 LLM call applies all per-game deltas to the canonical memory block.

No embedder, no stat pool, no question generation loop.
Store schema: {opponent_key: {"games_observed": int, "memories": list}}
"""

import json
import os
import re
import threading
from typing import Optional

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.tendency_prompts import (
    TENDENCY_REFLECTION_PROMPT,
    TENDENCY_APPLY_PROMPT,
    TENDENCY_STRATEGY_PROMPT,
    COOP_TENDENCY_STRATEGY_PROMPT,
    TENDENCY_INJECTION_BLOCK,
    TENDENCY_EMPTY_BLOCK,
)
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.utils.utils import strip_thinking_block


class SimpleTendencyAgent(PromptAgent):
    """
    Structured tendency agent: tracks opponent behavior as CONDITIONAL and INFERENCE
    memory entries, updated after each batch of games via two lightweight LLM calls.
    """

    # Class-level lock for thread-safe file logging
    _log_lock = threading.Lock()

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.tendency_store_path = getattr(config, "tendency_store_path", "tendency_store.json")
        self.batch_mode = getattr(config, "batch_mode", True)

        # tendency_store: {opponent_key: {"games_observed": int, "memories": list}}
        self.tendency_store: dict = {}
        if os.path.exists(self.tendency_store_path):
            self._load_store()

        # Per-game state
        self.current_opponent_key: Optional[str] = None
        self.current_game_intro: str = ""
        self._last_batch_result: Optional[dict] = None

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
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
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
    # Memory access helpers
    # ------------------------------------------------------------------

    def _get_memories(self) -> list:
        """Return the structured memories list for the active opponent."""
        if not self.current_opponent_key:
            return []
        entry = self.tendency_store.get(self.current_opponent_key, {})
        return entry.get("memories", [])

    def _get_games_observed(self) -> int:
        if not self.current_opponent_key:
            return 0
        return self.tendency_store.get(self.current_opponent_key, {}).get("games_observed", 0)



    # ------------------------------------------------------------------
    # Block rendering — pure Python, no LLM call
    # ------------------------------------------------------------------

    @staticmethod
    def _render_block(memories: list, games_observed: int) -> str:
        """Render structured memories to the rich display text injected in-game."""
        if not memories:
            return ""

        memories = SimpleTendencyAgent._sort_memories(memories)
        cond = [m for m in memories if m.get("type") == "CONDITIONAL"]
        inf_m = [m for m in memories if m.get("type") == "INFERENCE"]

        lines = [f"=== OPPONENT PROFILE (Based on {games_observed} games) ==="]

        if cond:
            lines.append(
                "\n[CONDITIONAL BEHAVIORS]\n"
                "When the trigger fires, the listed actions show the observed frequency distribution."
            )
            for m in cond:
                lines.append(f"\n► When: {m.get('situation') or m.get('game_state', '?')}")
                actions = m.get("actions", [])
                total = sum(a.get("count", 0) for a in actions)
                for a in actions:
                    pct = f"{a['count']}/{total}" if total else "0/?"
                    lines.append(f"    └── {a.get('description', '?')}  [{pct} times]")

        if inf_m:
            lines.append(
                "\n[INFERRED PRIVATE INFORMATION]\n"
                "When you observe the trigger action, the listed hidden states may explain it."
            )
            for m in inf_m:
                lines.append(f"\n► Observe: {m.get('public_action', '?')}")
                hi = m.get("hidden_information", [])
                total = sum(h.get("count", 0) for h in hi)
                for h in hi:
                    pct = f"{h['count']}/{total}" if total else "0/?"
                    lines.append(f"    → {h.get('description', '?')}  [{pct} times]")

        lines.append("\n===")
        return "\n".join(lines)

    @staticmethod
    def _sort_memories(memories: list) -> list:
        """Sort memories: CONDITIONAL first, then INFERENCE; descending by total count within each type."""
        type_order = {"CONDITIONAL": 0, "INFERENCE": 1}

        def _total_count(m: dict) -> int:
            variants = m.get("actions", m.get("hidden_information", []))
            return sum(v.get("count", 0) for v in variants)

        return sorted(
            memories,
            key=lambda m: (type_order.get(m.get("type"), 2), -_total_count(m)),
        )

    @staticmethod
    def _render_memory_stats(memories: list) -> str:
        """Render memories as pre-computed percentage statistics for the strategy prompt.
        All arithmetic is done here in Python — the strategy LLM receives ready-to-use numbers."""
        if not memories:
            return "(No memory entries yet.)"
        lines = []
        for m in memories:
            entry_type = m.get("type", "?")
            entry_id = m.get("id", "?")
            if entry_type == "CONDITIONAL":
                variants = m.get("actions", [])
                total = sum(v.get("count", 0) for v in variants)
                lines.append(f"[{entry_id}] CONDITIONAL | {total} total observations")
                lines.append(f"  Situation: {m.get('situation') or m.get('game_state', '?')}")
                for v in variants:
                    pct = round(100 * v["count"] / total) if total else 0
                    lines.append(f"  {v.get('id', '?')}. {v.get('description', '?')}  \u2192  {v['count']}/{total} ({pct}%)")
            elif entry_type == "INFERENCE":
                variants = m.get("hidden_information", [])
                total = sum(v.get("count", 0) for v in variants)
                lines.append(f"[{entry_id}] INFERENCE | {total} total observations")
                lines.append(f"  Public action: {m.get('public_action', '?')}")
                for v in variants:
                    pct = round(100 * v["count"] / total) if total else 0
                    lines.append(f"  {v.get('id', '?')}. {v.get('description', '?')}  \u2192  {v['count']}/{total} ({pct}%)")
            lines.append("")
        return "\n".join(lines).rstrip()

    @staticmethod
    def _stamp_last_batch(old_memories: list, new_memories: list, current_batch: int) -> list:
        """
        Stamp last_batch on entries that are new or changed vs old_memories.
        Unchanged entries retain their existing last_batch value.
        Purely Python — no LLM involvement; the LLM never sees this field.
        """
        old_by_id = {m["id"]: m for m in old_memories if "id" in m}
        for entry in new_memories:
            entry_id = entry.get("id")
            old_entry = old_by_id.get(entry_id)
            if old_entry is None:
                # New entry
                entry["last_batch"] = current_batch
            else:
                old_content = {k: v for k, v in old_entry.items() if k != "last_batch"}
                new_content = {k: v for k, v in entry.items() if k != "last_batch"}
                if old_content != new_content:
                    entry["last_batch"] = current_batch
                else:
                    # Unchanged — preserve existing last_batch (backward compat: default current_batch)
                    entry["last_batch"] = old_entry.get("last_batch", current_batch)
        return new_memories

    @staticmethod
    def _evict_stale_memories(memories: list, current_batch: int, stale_threshold: int = 5) -> list:
        """Remove whole entries not updated in the last `stale_threshold` consecutive batches."""
        return [
            m for m in memories
            if current_batch - m.get("last_batch", current_batch) < stale_threshold
        ]

    @staticmethod
    def _for_llm(memories: list) -> list:
        """Strip internal backend-only fields before serializing memories for any LLM prompt.
        The LLM should never see fields like `last_batch` — they are purely a backend concern."""
        _internal = {"last_batch"}
        return [{k: v for k, v in m.items() if k not in _internal} for m in memories]

    def _get_tendency_block(self) -> str:
        """Return the strategy brief for in-game injection.
        Falls back to TENDENCY_EMPTY_BLOCK when no strategy has been generated yet."""
        if not self.current_opponent_key:
            return TENDENCY_EMPTY_BLOCK
        entry = self.tendency_store.get(self.current_opponent_key, {})
        strategy = entry.get("strategy", "")
        return strategy if strategy else TENDENCY_EMPTY_BLOCK

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
    # JSON extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_array(text: str) -> Optional[list]:
        """
        Extract a JSON array from LLM output.
        Handles both fenced (```json ... ```) and raw output.
        Returns a list on success, None on failure.
        """
        text = text.strip()
        # Try fenced block first
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1)
        else:
            # Find outermost [ ... ]
            start = text.find('[')
            end = text.rfind(']')
            if start == -1 or end == -1 or end <= start:
                return None
            json_str = text[start:end + 1]

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # Logging
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

    # ------------------------------------------------------------------
    # Post-game: per-game reflection call
    # ------------------------------------------------------------------

    def _run_reflection(self, game_history: str) -> Optional[list]:
        """
        Single LLM call: given current memory + game trajectory, produce a JSON
        delta list. Retries once on JSON parse failure.

        Returns:
            list of delta dicts on success
            None if both attempts fail (this game's deltas will be skipped)
        """
        memories = self._get_memories()
        current_memories_str = json.dumps(self._for_llm(memories), indent=2) if memories else "[]"

        if self.agent_name:
            _my_label = f"{self.agent_name}_{self.player_id}" if getattr(self, 'player_id', None) else self.agent_name
            game_history = game_history.replace(_my_label, f"{_my_label} (You)")

        game_rules = getattr(self, "current_game_intro", "") or ""
        prompt = TENDENCY_REFLECTION_PROMPT.format(
            game_rules=game_rules,
            current_memories=current_memories_str,
            game_history=game_history,
        )
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(2):
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type="move")
            raw = responses[0]
            stripped = strip_thinking_block(raw)
            self._log_prompt(
                f"PER-GAME TENDENCY REFLECTION (attempt {attempt + 1})", prompt, raw
            )

            delta = self._extract_json_array(stripped)
            if delta is not None:
                return delta

            self.logger.warning(
                f"SimpleTendencyAgent: JSON parse failed on reflection attempt {attempt + 1}."
            )

        self.logger.warning(
            "SimpleTendencyAgent: reflection failed after 2 attempts; skipping this game's delta."
        )
        return None

    def post_game_update(
        self,
        game_history: str,
        final_board_state: str = "",
        env_name: str = "unknown",
    ):
        """
        Called by the framework after each game ends.

        Runs the per-game reflection call and stashes the delta (or None if
        reflection failed) in _last_batch_result for flush_batch_updates() to consume.
        """
        if not self.current_opponent_key:
            return

        self.logger.info(
            "-" * 20 + f" {self.agent_name} Post-Game Tendency Reflection " + "-" * 20
        )

        delta = self._run_reflection(game_history)

        if delta is not None:
            self.logger.info(f"Delta from this game:\n{json.dumps(delta, indent=2)}")
        else:
            self.logger.warning(
                "Delta is None (reflection failed); this game will be skipped in batch apply."
            )

        # Mark the agent's own moves in the game history before stashing
        marked_history = game_history
        if self.agent_name:
            _my_label = f"{self.agent_name}_{self.player_id}" if getattr(self, 'player_id', None) else self.agent_name
            marked_history = game_history.replace(_my_label, f"{_my_label} (You)")

        if self.batch_mode:
            self._last_batch_result = {
                "opponent_key": self.current_opponent_key,
                "delta": delta,
                "game_history": marked_history,
            }
            return

        # Non-batch fallback: apply delta immediately (single-game run)
        current_batch = self.tendency_store.get(
            self.current_opponent_key, {}
        ).get("current_batch", 0) + 1
        self._apply_deltas(self.current_opponent_key, [delta], num_games=1,
                           game_histories=[marked_history], current_batch=current_batch)

        self._save_store()

    # ------------------------------------------------------------------
    # Post-batch: apply all deltas to the canonical memory block
    # ------------------------------------------------------------------

    def _apply_deltas(
        self,
        opponent_key: str,
        deltas: list,
        num_games: int,
        game_histories: Optional[list] = None,
        current_batch: int = 0,
    ):
        """
        Single LLM call that applies all per-game delta lists to the existing
        memories, then calls _generate_strategy to synthesize the in-game playbook.

        deltas: list of per-game delta lists (each item is a list of delta dicts,
                or None if that game's reflection failed).
        game_histories: list of per-game trajectory strings (marked with "(You)").
        """
        entry = self.tendency_store.get(opponent_key, {})
        current_memories = entry.get("memories", [])
        games_observed = entry.get("games_observed", 0)

        # Skip LLM call entirely if there are no valid deltas
        valid_deltas = [d for d in deltas if d is not None]
        if not valid_deltas and not game_histories:
            self.logger.warning(
                f"SimpleTendencyAgent: all {num_games} game delta(s) for '{opponent_key}' "
                "were None and no histories; skipping apply call, incrementing games_observed only."
            )
            self.tendency_store[opponent_key] = {
                "games_observed": games_observed + num_games,
                "current_batch": current_batch,
                "memories": current_memories,
                "strategy": entry.get("strategy", ""),
            }
            return

        if not valid_deltas:
            self.logger.warning(
                f"SimpleTendencyAgent: all {num_games} game delta(s) for '{opponent_key}' "
                "were None; skipping apply call but still regenerating strategy."
            )
            new_memories = current_memories
        else:
            current_memories_str = json.dumps(self._for_llm(self._sort_memories(current_memories)), indent=2) if current_memories else "[]"

            # Format per-game deltas into numbered sections
            delta_sections = []
            for i, game_delta in enumerate(deltas, start=1):
                if game_delta is not None:
                    body = json.dumps(game_delta, indent=2)
                else:
                    body = "(Reflection failed for this game — skip.)"
                delta_sections.append(f"[Game {i}]\n{body}")
            all_deltas_str = "\n\n".join(delta_sections)

            game_rules = getattr(self, "current_game_intro", "") or ""
            prompt = TENDENCY_APPLY_PROMPT.format(
                game_rules=game_rules,
                current_memories=current_memories_str,
                all_deltas=all_deltas_str,
                num_games=num_games,
            )
            messages = [{"role": "user", "content": prompt}]
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type="move")
            raw = responses[0]
            stripped = strip_thinking_block(raw)

            self._log_prompt("BATCH TENDENCY APPLY", prompt, raw)

            new_memories = self._extract_json_array(stripped)
            if new_memories is None:
                self.logger.warning(
                    "SimpleTendencyAgent: failed to extract updated memories from apply response; "
                    "keeping existing memories."
                )
                new_memories = current_memories

        updated_games_observed = games_observed + num_games

        # Sort → stamp last_batch → evict stale entries (all pure Python, no LLM)
        sorted_new = self._sort_memories(new_memories)
        stamped = self._stamp_last_batch(current_memories, sorted_new, current_batch)
        new_memories = self._evict_stale_memories(stamped, current_batch)

        self.tendency_store[opponent_key] = {
            "games_observed": updated_games_observed,
            "current_batch": current_batch,
            "memories": new_memories,
            "strategy": entry.get("strategy", ""),  # placeholder; overwritten below
        }

        # Always regenerate strategy after every batch
        new_strategy = self._generate_strategy(opponent_key, game_histories or [], new_memories,
                                               updated_games_observed)
        self.tendency_store[opponent_key]["strategy"] = new_strategy

    @staticmethod
    def _is_cooperative_game(game_rules: str) -> bool:
        """Detect whether this is a cooperative game based on the game rules/intro text.
        Checks for known cooperative game identifiers (hanabi, cooperative negotiation)."""
        lower = game_rules.lower()
        return "hanabi" in lower or "cooperative" in lower

    def _generate_strategy(
        self,
        opponent_key: str,
        game_histories: list,
        memories: list,
        games_observed: int,
    ) -> str:
        """
        Post-batch LLM call: synthesizes the strategy brief from updated memories
        + all batch game trajectories + the previous strategy.
        Uses COOP_TENDENCY_STRATEGY_PROMPT for cooperative games (e.g. Hanabi),
        and TENDENCY_STRATEGY_PROMPT for competitive games.
        Returns the strategy as a plain-text string.
        """
        entry = self.tendency_store.get(opponent_key, {})
        previous_strategy = entry.get("strategy", "") or "(No previous strategy — first batch.)"

        sorted_mems = self._sort_memories(memories)
        memory_stats = self._render_memory_stats(sorted_mems)

        trajectory_sections = []
        for i, hist in enumerate(game_histories, start=1):
            body = hist.strip() if hist else "(No trajectory available.)"
            trajectory_sections.append(f"[Game {i}]\n{body}")
        all_trajectories = "\n\n".join(trajectory_sections) if trajectory_sections else "(None.)"

        game_rules = getattr(self, "current_game_intro", "") or ""

        if self._is_cooperative_game(game_rules):
            prompt_template = COOP_TENDENCY_STRATEGY_PROMPT
            phase_title = "COOPERATIVE STRATEGY SYNTHESIS"
        else:
            prompt_template = TENDENCY_STRATEGY_PROMPT
            phase_title = "STRATEGY SYNTHESIS"

        prompt = prompt_template.format(
            game_rules=game_rules,
            memory_stats=memory_stats,
            game_trajectories=all_trajectories,
            previous_strategy=previous_strategy,
            games_observed=games_observed,
            num_games=len(game_histories),
        )
        messages = [{"role": "user", "content": prompt}]
        responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type="move")
        raw = responses[0]
        stripped = strip_thinking_block(raw)

        self._log_prompt(phase_title, prompt, raw)
        return stripped.strip()

    def flush_batch_updates(self, gradient_data: list) -> None:
        """
        Called by main.py after all games in a batch complete.

        gradient_data: list of _last_batch_result dicts, one per game.
          Each dict: {"opponent_key": str, "delta": list | None}

        Groups deltas by opponent_key, then runs one apply LLM call per opponent key
        and writes the updated store to disk.
        """
        if not gradient_data:
            return

        # Group deltas AND game histories by opponent key
        grouped: dict[str, dict] = {}
        for item in gradient_data:
            if not isinstance(item, dict):
                continue
            key = item.get("opponent_key")
            delta = item.get("delta")
            game_history = item.get("game_history", "")
            if key:
                if key not in grouped:
                    grouped[key] = {"deltas": [], "game_histories": []}
                grouped[key]["deltas"].append(delta)
                grouped[key]["game_histories"].append(game_history)

        for opponent_key, data in grouped.items():
            deltas = data["deltas"]
            game_histories = data["game_histories"]
            valid_count = sum(1 for d in deltas if d is not None)
            # Increment per-opponent batch counter
            current_batch = self.tendency_store.get(opponent_key, {}).get("current_batch", 0) + 1
            self.logger.info(
                f"SimpleTendencyAgent flush: applying {len(deltas)} game delta(s) "
                f"for opponent '{opponent_key}' ({valid_count} valid, "
                f"{len(deltas) - valid_count} skipped) — batch #{current_batch}"
            )
            self._apply_deltas(opponent_key, deltas, num_games=len(deltas),
                               game_histories=game_histories, current_batch=current_batch)

        self._save_store()
