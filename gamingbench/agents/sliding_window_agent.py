import os
import re
from typing import Tuple

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.sw.sw_store import SlidingWindowStore
from gamingbench.prompts.sliding_window_prompts import SW_INJECTION_PROMPT, SW_UPDATE_PROMPT


class SlidingWindowAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(SlidingWindowAgent, self).__init__(config, **kwargs)
        
        self.sw_store_path = getattr(self, "sw_store_path", "sw_store.json")
        self.sw_store = SlidingWindowStore()
        
        if os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)
            
        self.current_game_name = None
        
        # Batch mode variables
        self.batch_mode: bool = False
        self._last_batch_result = None  # Stores raw game_history string
        
        self.in_game_obs_mode: bool = False
        self.in_game_obs: str | None = None
        
        self.sw_anti_decay: bool = False
        self.sw_anti_contradiction: bool = False
        
        # Memory mode: 'default' | 'reputation' | 'strategy' | 'dual' | 'dual-structured' | 'structured'
        self.sw_memory_mode: str = getattr(self, 'sw_memory_mode', 'default')
        
        # Enable per-observation frequency tracking in the memory update prompt
        self.sw_track_frequency: bool = False
        
        # Trajectory selection mode: 'best' | 'worst' | 'random' | '' (disabled)
        # When set, main.py plays batch_size games but only selects 8 for memory update
        self.sw_trajectory_select: str = ''
        
        # When True, trajectory selection balances across roles (e.g. 4 Seller + 4 Buyer)
        self.sw_trajectory_balanced: bool = False
        
        # Explore mode (dual only): best-strategy version tracking + guided exploration
        # in the strategy update prompt, ported from SimpleTendencyAgent st_use_explore.
        self.sw_explore_strategy: bool = False
        
        # Deterministic exploration mode: detects plateaus using a patience window
        self.sw_deterministic_explore: bool = False
        self.best_historical_avg: float = float('-inf')
        self.best_strategy_text: str = ""
        self.patience_counter: int = 0
        self.exploration_mode_active: bool = False

    def set_storage_dir(self, storage_dir):
        """Called by main.py to align SW storage with the run's experiment folder."""
        base = os.path.basename(self.sw_store_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
                
        self.sw_store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking."""
        if not self.batch_mode and os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)
            
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        self.game_count += 1
        
        self.in_game_obs = None
        
        # We don't care about opponent_key for SW, but we need the game name.
        # Since game_intro doesn't strictly have game_name isolated easily, 
        # we'll extract it from the env_name in step or post_game_update.

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        obs_env = observations['env_name']
        env_name = getattr(self, 'current_game_name', obs_env)
        if not env_name or env_name == 'unknown':
            env_name = obs_env
            
        self.current_game_name = env_name
        
        from gamingbench.prompts.observation_prompts import construct_game_intro
        game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
        
        mode = getattr(self, 'sw_memory_mode', 'default')
        
        if mode == 'reputation':
            from gamingbench.prompts.sliding_window_prompts import SW_REPUTATION_INJECTION_PROMPT
            notes = self.sw_store.get(f"{env_name}_reputation") or self.sw_store.get(env_name)
            if notes:
                sw_injection = SW_REPUTATION_INJECTION_PROMPT.format(
                    game_name=env_name,
                    notes_text=notes
                )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
                
        elif mode == 'strategy':
            from gamingbench.prompts.sliding_window_prompts import SW_STRATEGY_INJECTION_PROMPT
            notes = self.sw_store.get(f"{env_name}_strategy") or self.sw_store.get(env_name)
            if notes:
                sw_injection = SW_STRATEGY_INJECTION_PROMPT.format(
                    game_name=env_name,
                    notes_text=notes
                )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
                
        elif mode == 'dual':
            from gamingbench.prompts.sliding_window_prompts import SW_DUAL_INJECTION_PROMPT, SW_REPUTATION_INJECTION_PROMPT, SW_STRATEGY_INJECTION_PROMPT
            reputation = self.sw_store.get(f"{env_name}_reputation")
            strategy = self.sw_store.get(f"{env_name}_strategy")
            if reputation and strategy:
                sw_injection = SW_DUAL_INJECTION_PROMPT.format(
                    game_name=env_name,
                    reputation_text=reputation,
                    strategy_text=strategy
                )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
            elif reputation:
                sw_injection = SW_REPUTATION_INJECTION_PROMPT.format(game_name=env_name, notes_text=reputation)
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
            elif strategy:
                sw_injection = SW_STRATEGY_INJECTION_PROMPT.format(game_name=env_name, notes_text=strategy)
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
                
        elif mode == 'dual-structured':
            from gamingbench.prompts.sliding_window_prompts import SW_DUAL_STRUCTURED_INJECTION_PROMPT
            strategy = self.sw_store.get(f"{env_name}_strategy")
            if strategy:
                sw_injection = SW_DUAL_STRUCTURED_INJECTION_PROMPT.format(
                    game_name=env_name,
                    notes_text=strategy
                )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
                
        elif mode == 'structured':
            from gamingbench.prompts.sliding_window_prompts import SW_STRUCTURED_INJECTION_PROMPT
            strategy = self.sw_store.get(f"{env_name}_strategy")
            if strategy:
                sw_injection = SW_STRUCTURED_INJECTION_PROMPT.format(
                    game_name=env_name,
                    notes_text=strategy
                )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
                
        else:  # mode == 'default' (original bullet-list reputation)
            current_notes = self.sw_store.get(env_name)
            if not current_notes and obs_env != env_name:
                current_notes = self.sw_store.get(obs_env)
            if current_notes:
                # Use best_worst injection prompt if trajectory select is best_worst
                if getattr(self, 'sw_trajectory_select', '') == 'best_worst':
                    from gamingbench.prompts.sliding_window_prompts import SW_BEST_WORST_INJECTION_PROMPT
                    sw_injection = SW_BEST_WORST_INJECTION_PROMPT.format(
                        game_name=env_name,
                        notes_text=current_notes
                    )
                else:
                    from gamingbench.prompts.sliding_window_prompts import SW_INJECTION_PROMPT
                    sw_injection = SW_INJECTION_PROMPT.format(
                        game_name=env_name,
                        notes_text=current_notes
                    )
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
            
        if getattr(self, 'in_game_obs_mode', False) and getattr(self, 'in_game_obs', None):
            from gamingbench.prompts.sliding_window_prompts import SW_OBS_INJECTION_PROMPT
            obs_block = SW_OBS_INJECTION_PROMPT.format(in_game_obs=self.in_game_obs)
            # Append after the last injection block already added, or after game_intro
            # We insert after game_intro region (which now may include the sw_injection)
            target = game_intro
            # Try to find a more specific anchor if sw_injection was added
            try:
                if '=== END OPPONENT REPUTATION NOTE ===' in observation_prompt:
                    target_marker = '=== END OPPONENT REPUTATION NOTE ==='
                    observation_prompt = observation_prompt.replace(
                        target_marker, target_marker + "\n\n" + obs_block, 1)
                elif '=== END STRATEGY NOTE ===' in observation_prompt:
                    target_marker = '=== END STRATEGY NOTE ==='
                    observation_prompt = observation_prompt.replace(
                        target_marker, target_marker + "\n\n" + obs_block, 1)
                else:
                    observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + obs_block, 1)
            except Exception:
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + obs_block, 1)
            
        return system_prompt, observation_prompt

    def step(self, observations):
        original_constructor = self.step_prompt_constructor
        
        def wrapped_constructor(obs):
            instruct = original_constructor(obs).copy()
            if getattr(self, 'in_game_obs_mode', False):
                from gamingbench.prompts.sliding_window_prompts import SW_OBS_GENERATION_SUFFIX
                instruct['prompt'] += "\n\n" + SW_OBS_GENERATION_SUFFIX
            return instruct
            
        self.step_prompt_constructor = wrapped_constructor
        
        original_llm_query = self.llm_query
        
        def wrapped_llm_query(messages, n=1, stop=None, prompt_type='move'):
            responses, query = original_llm_query(messages, n, stop, prompt_type)
            if getattr(self, 'in_game_obs_mode', False):
                import re
                from gamingbench.utils.utils import strip_thinking_block
                clean_response = strip_thinking_block(responses[0])
                match = re.search(r'<obs>(.*?)</obs>', clean_response, re.DOTALL)
                if match:
                    self.in_game_obs = match.group(1).strip()
                    self.logger.info(f"[In-Game Obs] Parsed:\n{self.in_game_obs}")
            return responses, query
            
        self.llm_query = wrapped_llm_query
        
        try:
            return super().step(observations)
        finally:
            self.step_prompt_constructor = original_constructor
            self.llm_query = original_llm_query
    def chat_step(self, observations, chat_history_str: str):
        original_llm_query = self.llm_query
        
        def wrapped_llm_query(messages, n=1, stop=None, prompt_type='move'):
            if getattr(self, 'in_game_obs_mode', False):
                from gamingbench.prompts.sliding_window_prompts import SW_OBS_GENERATION_SUFFIX
                messages[-1]['content'] += "\n\n" + SW_OBS_GENERATION_SUFFIX
                
            responses, query = original_llm_query(messages, n, stop, prompt_type)
            if getattr(self, 'in_game_obs_mode', False):
                import re
                from gamingbench.utils.utils import strip_thinking_block
                clean_response = strip_thinking_block(responses[0])
                match = re.search(r'<obs>(.*?)</obs>', clean_response, re.DOTALL)
                if match:
                    self.in_game_obs = match.group(1).strip()
                    self.logger.info(f"[In-Game Obs] Parsed from chat:\n{self.in_game_obs}")
            return responses, query
            
        self.llm_query = wrapped_llm_query
        
        try:
            return super().chat_step(observations, chat_history_str)
        finally:
            self.llm_query = original_llm_query

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called at the end of the game."""
        self.current_game_name = env_name
        
        if getattr(self, 'in_game_obs_mode', False) and getattr(self, 'in_game_obs', None):
            game_history = self._refine_final_obs(game_history)
            
        if self.batch_mode:
            self._last_batch_result = game_history
            self.logger.info('Batch mode: SlidingWindowAgent stored result, synthesis deferred.')
            return
            
        # If not batch mode, just do it directly.
        self.flush_batch_updates([game_history])

    def _refine_final_obs(self, game_history: str) -> str:
        """Post-game: refine the running in-game obs using the full trajectory.
        
        Result is logged to sw_store_trace.log and returned as the
        'game_history' string passed to flush_batch_updates.
        """
        from gamingbench.prompts.sliding_window_prompts import SW_OBS_FINAL_REFINEMENT_PROMPT
        from gamingbench.utils.utils import strip_thinking_block
        
        prompt = SW_OBS_FINAL_REFINEMENT_PROMPT.format(
            game_name=self.current_game_name or "unknown",
            in_game_obs=self.in_game_obs,
            game_history=game_history
        )
        messages = [{"role": "user", "content": prompt}]
        log_file = self.sw_store_path.replace('.json', '_trace.log')
        
        try:
            generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            refined = strip_thinking_block(generations[0]).strip()
            self.logger.info(f"[In-Game Obs] Final refined observation:\n{refined}")
        except Exception as e:
            self.logger.error(f"Failed to refine final observation: {e}")
            refined = self.in_game_obs
        
        with open(log_file, 'a') as f:
            f.write('=== IN-GAME OBS FINAL REFINEMENT ===\n')
            f.write(f'REFINEMENT PROMPT:\n{prompt}\n\n')
            f.write(f'RUNNING OBS (pre-refinement):\n{self.in_game_obs}\n\n')
            f.write(f'REFINED OBS:\n{refined}\n')
            f.write('=' * 50 + '\n\n')
        
        return refined

    @staticmethod
    def _parse_numbered_observations(notes_text: str) -> list:
        """Parse the current memory text into a list of {id, text, count} dicts."""
        import re
        observations = []
        obs_id = 0
        
        for line in notes_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            m = re.match(r'^(\d+)\.\s*(.+)', line)
            if m:
                obs_id = int(m.group(1))
                rest = m.group(2)
            else:
                m = re.match(r'^[*\-•]\s*(.+)', line)
                if m:
                    obs_id += 1
                    rest = m.group(1)
                else:
                    continue
                    
            count_match = re.search(r'\[count:\s*(\d+)\]', rest)
            if count_match:
                count = int(count_match.group(1))
                text = rest[:count_match.start()].strip()
            else:
                count = 1
                text = rest.strip()
                
            observations.append({"id": obs_id, "text": text, "count": count})
        return observations

    @staticmethod
    def _render_observations(observations: list) -> str:
        """Render observations back to numbered text format with counts."""
        return '\n'.join(f"{obs['id']}. {obs['text']} [count: {obs['count']}]" for obs in observations)

    @staticmethod
    def _apply_freq_v2_update(observations: list, update: dict) -> list:
        obs_map = {o["id"]: o for o in observations}
        
        game_evals = update.get("game_evaluations", [])
        if isinstance(game_evals, list):
            for game_eval in game_evals:
                if isinstance(game_eval, dict):
                    supported = game_eval.get("supported_ids", [])
                    if isinstance(supported, list):
                        for obs_id in supported:
                            if isinstance(obs_id, int) and obs_id in obs_map:
                                obs_map[obs_id]["count"] += 1
                                
        reworded = update.get("reworded", {})
        if isinstance(reworded, dict):
            for obs_id_str, new_text in reworded.items():
                if isinstance(new_text, str):
                    try:
                        obs_id = int(obs_id_str)
                        if obs_id in obs_map:
                            obs_map[obs_id]["text"] = new_text.strip()
                    except (ValueError, TypeError):
                        pass
                
        new_obs = update.get("new_observations", [])
        if isinstance(new_obs, list):
            next_id = max((o["id"] for o in observations), default=0) + 1
            for new_text in new_obs:
                if isinstance(new_text, str):
                    new_text = new_text.strip()
                    if new_text:
                        observations.append({"id": next_id, "text": new_text, "count": 1})
                        next_id += 1
                
        return observations

    @staticmethod
    def _parse_freq_v2_json(text: str) -> dict | None:
        import json, re
        text = text.strip()
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                return None
            json_str = text[start:end + 1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    # ------------------------------------------------------------------
    # Explore mode: strategy version tracking (ported from SimpleTendencyAgent)
    # ------------------------------------------------------------------

    _NO_BEST_STRATEGY = "(No best strategy yet — first version.)"

    def _get_explore_state(self, env_name: str) -> dict:
        """Return (creating if needed) the persisted explore-mode state for env_name."""
        key = f"{env_name}_explore"
        state = self.sw_store.get(key)
        if not isinstance(state, dict):
            state = {}
            self.sw_store.update(key, state)
        return state

    def _update_version_tracking(self, env_name: str, utilities: list) -> None:
        """Accumulate utility for the current strategy version; promote to best if it outperforms.

        Must run BEFORE the strategy is regenerated, since `utilities` were earned
        by the strategy that was active during this batch.
        """
        state = self._get_explore_state(env_name)
        valid = [float(u) for u in utilities if u is not None]
        if not valid:
            return

        state["current_total_utility"] = state.get("current_total_utility", 0.0) + sum(valid)
        state["current_games_played"] = state.get("current_games_played", 0) + len(valid)
        current_avg = state["current_total_utility"] / state["current_games_played"]

        best_avg = state.get("best_avg_utility", float("-inf"))
        is_best_baseline = state.get("is_best_baseline", False)

        if current_avg > best_avg:
            is_best_baseline = True
            self.logger.info(
                f"[Explore] Version tracking: promoted current to best "
                f"(avg={current_avg:.2f}, {state['current_games_played']} games)"
            )

        # If this is the best baseline (just promoted, or inherited), keep best stats synced
        if is_best_baseline:
            state["best_strategy"] = self.sw_store.get(f"{env_name}_strategy") or ""
            state["best_avg_utility"] = current_avg
            state["best_total_utility"] = state["current_total_utility"]
            state["best_games_played"] = state["current_games_played"]

        state["is_best_baseline"] = is_best_baseline

        self.logger.info(
            f"[Explore] Version tracking: current avg={current_avg:.2f} "
            f"({state['current_games_played']} games), "
            f"best avg={state.get('best_avg_utility', 0.0):.2f} "
            f"({state.get('best_games_played', 0)} games), batch utilities={valid}"
        )

    def _explore_prompt_kwargs(self, env_name: str) -> dict:
        """Template kwargs for SW_UPDATE_PROMPT_DUAL_STRATEGY_EXPLORE."""
        state = self._get_explore_state(env_name)
        current_gp = state.get("current_games_played", 0)
        return dict(
            best_strategy=state.get("best_strategy") or self._NO_BEST_STRATEGY,
            best_avg_utility=state.get("best_avg_utility", 0.0),
            best_games_played=state.get("best_games_played", 0),
            current_avg_utility=(state.get("current_total_utility", 0.0) / current_gp
                                 if current_gp > 0 else 0.0),
            current_games_played=current_gp,
        )

    _INHERIT_BEST_RE = re.compile(r'^\s*[*_`#>\s]*\[INHERIT BEST\][*_`]*\s*')

    def _apply_explore_decision(self, env_name: str, new_strategy: str, reputation_notes: str, formatted_histories: str) -> Tuple[str, str, str, str]:
        """Parse [INHERIT BEST], and if EVOLVE, audit the draft strategy against the best baseline."""
        state = self._get_explore_state(env_name)
        m = self._INHERIT_BEST_RE.match(new_strategy)
        if m:
            cleaned = new_strategy[m.end():].strip()
            # Restore / preserve the best baseline counters
            state["current_total_utility"] = state.get("best_total_utility", state.get("current_total_utility", 0.0))
            state["current_games_played"] = state.get("best_games_played", state.get("current_games_played", 0))
            state["is_best_baseline"] = True
            self.logger.info("[Explore] Strategy synthesis: [INHERIT BEST] — restoring/preserving best baseline.")
            return cleaned, "INHERIT_BEST", "", ""

        # Attempt to EVOLVE, but audit it first
        best_strat = state.get("best_strategy", "")
        if not best_strat or best_strat == self._NO_BEST_STRATEGY:
            # If there's no baseline to compare to, just accept EVOLVE immediately
            state["current_total_utility"] = 0.0
            state["current_games_played"] = 0
            state["is_best_baseline"] = False
            self.logger.info("[Explore] Strategy synthesis: EVOLVE — no baseline to audit against, resetting counters.")
            return new_strategy.strip(), "EVOLVE", "", "No baseline strategy to audit against."

        # Perform Audit
        from gamingbench.prompts.sliding_window_prompts import SW_AUDIT_STRATEGY_PROMPT
        n_games = len([h for h in formatted_histories.split('===') if 'Game' in h]) // 2
        
        audit_prompt = SW_AUDIT_STRATEGY_PROMPT.format(
            game_name=env_name,
            game_intro=getattr(self, 'current_game_intro', "Game rules unavailable."),
            reputation_notes=reputation_notes,
            n=n_games,
            game_histories=formatted_histories,
            strategy_a=best_strat,
            strategy_b=new_strategy.strip()
        )
        
        try:
            generations, _ = self.llm_query([{"role": "user", "content": audit_prompt}], n=1, stop=None, prompt_type='move')
            raw_audit = generations[0]
        except Exception as e:
            self.logger.error(f"[Explore] Audit LLM call failed: {e}")
            raw_audit = "AVG_STRATEGY_A_SCORE: 0\\nAVG_STRATEGY_B_SCORE: 0\\nError during audit."

        # Parse scores
        import re
        avg_a = 0.0
        avg_b = 0.0
        match_a = re.search(r"AVG_STRATEGY_A_SCORE:\s*([\d\.]+)", raw_audit)
        match_b = re.search(r"AVG_STRATEGY_B_SCORE:\s*([\d\.]+)", raw_audit)
        if match_a:
            try: avg_a = float(match_a.group(1))
            except ValueError: pass
        if match_b:
            try: avg_b = float(match_b.group(1))
            except ValueError: pass
            
        if avg_b > avg_a:
            # Audit approved EVOLVE
            state["current_total_utility"] = 0.0
            state["current_games_played"] = 0
            state["is_best_baseline"] = False
            self.logger.info(f"[Explore] Auditor APPROVED Evolve. A={avg_a}, B={avg_b}")
            return new_strategy.strip(), "EVOLVE", audit_prompt, raw_audit
        else:
            # Audit rejected EVOLVE, override to INHERIT BEST
            self.logger.info(f"[Explore] Auditor REJECTED Evolve. A={avg_a}, B={avg_b}. Forcing INHERIT_BEST.")
            state["current_total_utility"] = state.get("best_total_utility", state.get("current_total_utility", 0.0))
            state["current_games_played"] = state.get("best_games_played", state.get("current_games_played", 0))
            state["is_best_baseline"] = True
            return best_strat, "INHERIT_BEST", audit_prompt, raw_audit

    def flush_batch_updates(self, game_histories: list, utilities: list | None = None) -> None:
        """Perform a single unified SW update from a completed batch of N games.

        utilities: per-game scores for ALL games in the batch (only used when
        sw_explore_strategy is enabled, for strategy version tracking).
        """
        if not game_histories:
            return
            
        n = len(game_histories)
        self.logger.info(f'-' * 20 + f'Batch SW Flush (N={n})' + '-' * 20)
        
        env_name = self.current_game_name or "unknown"
        
        # --- Deterministic Exploration Plateau Detection ---
        was_exploring = getattr(self, 'exploration_mode_active', False)
        explore_success = False
        
        if getattr(self, 'sw_deterministic_explore', False) and utilities:
            avg_b = sum(utilities) / len(utilities)
            
            if was_exploring:
                self.exploration_mode_active = False
                
            if avg_b > self.best_historical_avg:
                if was_exploring:
                    explore_success = True
                self.best_historical_avg = avg_b
                self.best_strategy_text = self.sw_store.get(f"{env_name}_strategy") or "(No strategy yet)"
                self.patience_counter = 0
                self.exploration_mode_active = False
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= 3:
                self.exploration_mode_active = True
                self.patience_counter = 0
                
            self.logger.info(f"[DeterministicExplore] Batch avg_b={avg_b:.2f}, best={self.best_historical_avg:.2f}, patience={self.patience_counter}, explore_active={self.exploration_mode_active}, was_exploring={was_exploring}, success={explore_success}")
        
        mode = getattr(self, 'sw_memory_mode', 'default')
        explore = getattr(self, 'sw_explore_strategy', False) and mode in ('dual', 'strategy')
        
        # Dynamically set chunk size to process all provided histories in a single chunk
        # if the user explicitly requested a larger block size (e.g., incl 12)
        chunk_size = max(8, len(game_histories))
        chunks = [game_histories[i:i + chunk_size] for i in range(0, len(game_histories), chunk_size)]
        
        # --- Helper to pick the right single-memory update prompt ---
        def _get_single_prompt_template(for_mode):
            """Returns the prompt template for single-memory modes (default, reputation, strategy)."""
            if for_mode == 'reputation':
                if getattr(self, 'sw_anti_decay', False):
                    from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_REPUTATION_ANTI_DECAY
                    template = SW_UPDATE_PROMPT_REPUTATION_ANTI_DECAY
                elif getattr(self, 'sw_anti_contradiction', False):
                    from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_REPUTATION_ANTI_CONTRADICTION
                    template = SW_UPDATE_PROMPT_REPUTATION_ANTI_CONTRADICTION
                else:
                    from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_REPUTATION
                    template = SW_UPDATE_PROMPT_REPUTATION
            elif for_mode == 'strategy':
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_STRATEGY
                template = SW_UPDATE_PROMPT_STRATEGY
            else:  # 'default'
                if getattr(self, 'in_game_obs_mode', False):
                    if getattr(self, 'sw_anti_decay', False):
                        from gamingbench.prompts.sliding_window_prompts import SW_OBS_UPDATE_PROMPT_ANTI_DECAY
                        template = SW_OBS_UPDATE_PROMPT_ANTI_DECAY
                    elif getattr(self, 'sw_anti_contradiction', False):
                        from gamingbench.prompts.sliding_window_prompts import SW_OBS_UPDATE_PROMPT_ANTI_CONTRADICTION
                        template = SW_OBS_UPDATE_PROMPT_ANTI_CONTRADICTION
                    else:
                        from gamingbench.prompts.sliding_window_prompts import SW_OBS_UPDATE_PROMPT
                        template = SW_OBS_UPDATE_PROMPT
                else:
                    if getattr(self, 'sw_anti_decay', False):
                        from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_ANTI_DECAY
                        template = SW_UPDATE_PROMPT_ANTI_DECAY
                    elif getattr(self, 'sw_anti_contradiction', False):
                        from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_ANTI_CONTRADICTION
                        template = SW_UPDATE_PROMPT_ANTI_CONTRADICTION
                    else:
                        from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT
                        template = SW_UPDATE_PROMPT
            if getattr(self, 'sw_track_frequency', False):
                from gamingbench.prompts.sliding_window_prompts import SW_TRACK_FREQUENCY_SUFFIX
                template = template + "\n" + SW_TRACK_FREQUENCY_SUFFIX
            return template
        
        # --- Helper to pick reputation update prompt for dual modes ---
        def _get_dual_reputation_prompt():
            if getattr(self, 'sw_anti_decay', False):
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_DECAY
                template = SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_DECAY
            elif getattr(self, 'sw_anti_contradiction', False):
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_CONTRADICTION
                template = SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_CONTRADICTION
            else:
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_REPUTATION
                template = SW_UPDATE_PROMPT_DUAL_REPUTATION
            if getattr(self, 'sw_track_frequency', False):
                from gamingbench.prompts.sliding_window_prompts import SW_TRACK_FREQUENCY_SUFFIX
                template = template + "\n" + SW_TRACK_FREQUENCY_SUFFIX
            return template
        
        log_file = self.sw_store_path.replace('.json', '_trace.log')
        game_intro = getattr(self, 'current_game_intro', "Game rules unavailable.")
        
        from gamingbench.utils.utils import strip_thinking_block
        
        if mode in ('dual', 'dual-structured', 'structured'):
            # ----------------------------------------------------------------
            # Dual-memory flow: two sequential LLM calls per chunk
            # Call 1: update reputation  (anti-decay/contradiction apply here)
            # Call 2: update strategy    (always free incremental refinement)
            # ----------------------------------------------------------------
            reputation_key = f"{env_name}_reputation"
            strategy_key = f"{env_name}_strategy"
            
            current_reputation = self.sw_store.get(reputation_key) or ""
            rep_obs_list = None
            if getattr(self, 'sw_track_frequency', False):
                rep_obs_list = self._parse_numbered_observations(current_reputation)
                current_reputation = self._render_observations(rep_obs_list) if rep_obs_list else "(No observations yet)"
            elif not current_reputation:
                current_reputation = "(No reputation note yet)"
            
            current_strategy = self.sw_store.get(strategy_key) or "(No strategy yet)"
            
            rep_prompt_template = _get_dual_reputation_prompt()
            
            if mode == 'dual-structured':
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_STRATEGY_STRUCTURED as strat_prompt_template
            elif mode == 'structured':
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_STRATEGY_STRUCTURED as strat_prompt_template
            else:
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_STRATEGY as strat_prompt_template
            
            if explore:
                from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_DUAL_STRATEGY_EXPLORE
                # Version tracking must run before the strategy is regenerated. Batch 1 is
                # played zero-shot (no strategy yet), so its utility is discarded to avoid
                # locking in an empty baseline.
                if self.sw_store.get(strategy_key) and utilities:
                    self._update_version_tracking(env_name, utilities)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_n = len(chunk)
                formatted_histories = ""
                for i, h in enumerate(chunk):
                    formatted_histories += f"=== Game {chunk_idx * chunk_size + i + 1} ===\n{h}\n\n"
                
                # --- Call 1: Reputation update ---
                rep_prompt = rep_prompt_template.format(
                    game_name=env_name,
                    game_intro=game_intro,
                    n=chunk_n,
                    old_notes=current_reputation,
                    game_histories=formatted_histories
                )
                
                if getattr(self, 'sw_deterministic_explore', False) and was_exploring:
                    from gamingbench.prompts.sliding_window_prompts import SW_DETERMINISTIC_EXPLORE_REP_CONTEXT
                    rep_explore_context = SW_DETERMINISTIC_EXPLORE_REP_CONTEXT.format(wild_strategy=current_strategy)
                    rep_prompt = rep_explore_context + "\n\n" + rep_prompt
                    
                messages = [{"role": "user", "content": rep_prompt}]
                rep_raw = ""
                try:
                    generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                    rep_raw = generations[0]
                    raw_reputation = strip_thinking_block(rep_raw)
                    
                    if getattr(self, 'sw_track_frequency', False):
                        update_json = self._parse_freq_v2_json(raw_reputation)
                        if update_json is not None:
                            rep_obs_list = self._apply_freq_v2_update(rep_obs_list, update_json)
                            new_reputation = self._render_observations(rep_obs_list)
                        else:
                            self.logger.warning(f"[Dual] Failed to parse freq v2 json for chunk {chunk_idx}, ignoring chunk.")
                            new_reputation = current_reputation
                    else:
                        new_reputation = raw_reputation
                except Exception as e:
                    self.logger.error(f"[Dual] Reputation update failed for chunk {chunk_idx}: {e}")
                    new_reputation = current_reputation
                
                self.logger.info(f'[Dual] Chunk {chunk_idx} Reputation (N={chunk_n}):\n{new_reputation}')
                with open(log_file, 'a') as f:
                    f.write(f'=== CHUNK {chunk_idx} DUAL REPUTATION UPDATE (N={chunk_n}) ===\n')
                    f.write(f'PROMPT:\n{rep_prompt}\n')
                    f.write(f'RAW THINKING:\n{self._extract_thinking(rep_raw)}\n')
                    f.write(f'NEW REPUTATION:\n{new_reputation}\n')
                    f.write('=' * 50 + '\n\n')
                
                # --- Call 2: Strategy update (uses NEW reputation from Call 1) ---
                # Compute total games observed for structured format header
                games_observed = getattr(self, 'game_count', chunk_n)
                
                strat_kwargs = dict(
                    game_name=env_name,
                    game_intro=game_intro,
                    n=chunk_n,
                    reputation_notes=new_reputation,
                    old_strategy=current_strategy,
                    game_histories=formatted_histories
                )
                # dual-structured and structured need games_observed for the output header
                if mode in ('dual-structured', 'structured'):
                    strat_kwargs['games_observed'] = games_observed
                
                # Explore: only the LAST chunk gets the explore template + decision, so there
                # is exactly one EVOLVE / INHERIT BEST decision per batch.
                is_explore_call = explore and chunk_idx == len(chunks) - 1
                if is_explore_call:
                    strat_kwargs.update(self._explore_prompt_kwargs(env_name))
                    strat_prompt = SW_UPDATE_PROMPT_DUAL_STRATEGY_EXPLORE.format(**strat_kwargs)
                else:
                    strat_prompt = strat_prompt_template.format(**strat_kwargs)
                    
                strategy_ok = False
                strat_raw = ""
                
                # If we are in deterministic explore and the explore batch failed, skip the LLM call to save tokens and prevent regression
                if getattr(self, 'sw_deterministic_explore', False) and was_exploring and not explore_success:
                    self.logger.info(f"[Dual] Exploration failed. Skipping strategy LLM call and rolling back to best strategy.")
                    new_strategy = self.best_strategy_text
                    strategy_ok = True
                else:
                    if getattr(self, 'sw_deterministic_explore', False):
                        best_block = f"\n--- BEST PERFORMING STRATEGY (High Score Anchor) ---\n{self.best_strategy_text or '(No best strategy established yet)'}\n"
                        if "--- GAME HISTORIES" in strat_prompt:
                            strat_prompt = strat_prompt.replace("--- GAME HISTORIES", best_block + "\n--- GAME HISTORIES", 1)
                        else:
                            strat_prompt += "\n" + best_block
                            
                        if getattr(self, 'exploration_mode_active', False):
                            from gamingbench.prompts.sliding_window_prompts import SW_DETERMINISTIC_EXPLORE_INJECTION
                            strat_prompt += SW_DETERMINISTIC_EXPLORE_INJECTION

                    messages = [{"role": "user", "content": strat_prompt}]
                    try:
                        generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                        strat_raw = generations[0]
                        new_strategy = strip_thinking_block(strat_raw)
                        strategy_ok = True
                    except Exception as e:
                        self.logger.error(f"[Dual] Strategy update failed for chunk {chunk_idx}: {e}")
                        new_strategy = current_strategy
                
                decision = None
                raw_audit = ""
                audit_prompt = ""
                if is_explore_call and strategy_ok:
                    new_strategy, decision, audit_prompt, raw_audit = self._apply_explore_decision(env_name, new_strategy, current_reputation, formatted_histories)
                
                self.logger.info(f'[Dual] Chunk {chunk_idx} Strategy (N={chunk_n}):\\n{new_strategy}')
                with open(log_file, 'a') as f:
                    f.write(f'=== CHUNK {chunk_idx} DUAL STRATEGY UPDATE (N={chunk_n}) ===\\n')
                    f.write(f'PROMPT:\\n{strat_prompt}\\n')
                    f.write(f'RAW THINKING:\\n{self._extract_thinking(strat_raw)}\\n')
                    f.write(f'NEW STRATEGY:\\n{new_strategy}\\n')
                    f.write('=' * 50 + '\\n\\n')
                    if is_explore_call:
                        state = self._get_explore_state(env_name)
                        f.write('=== EXPLORE DECISION ===\\n')
                        f.write(f'DECISION: {decision or "SKIPPED (strategy call failed)"}\\n')
                        if audit_prompt:
                            f.write(f'AUDIT PROMPT:\\n{audit_prompt}\\n\\n')
                        if raw_audit:
                            f.write(f'AUDIT LOG:\\n{raw_audit}\\n\\n')
                        f.write(f'BATCH UTILITIES: {utilities}\\n')
                        f.write(f'CURRENT: total={state.get("current_total_utility", 0.0)}, '
                                f'games={state.get("current_games_played", 0)}\\n')
                        f.write(f'BEST: avg={state.get("best_avg_utility", None)}, '
                                f'games={state.get("best_games_played", 0)}\\n')
                        f.write('=' * 50 + '\\n\\n')
                
                # Carry forward for next chunk
                current_reputation = new_reputation
                current_strategy = new_strategy
            
            # Save both memories
            if getattr(self, 'sw_deterministic_explore', False) and was_exploring and explore_success:
                self.logger.info("[DeterministicExplore] Exploration succeeded! Adopted wild strategy as new baseline.")
                
            self.sw_store.update(reputation_key, current_reputation)
            self.sw_store.update(strategy_key, current_strategy)
            self.sw_store.save(self.sw_store_path)
            
        else:
            # ----------------------------------------------------------------
            # Single-memory flow: one LLM call per chunk
            # Modes: 'default', 'reputation', 'strategy'
            # ----------------------------------------------------------------
            if mode == 'reputation':
                store_key = f"{env_name}_reputation"
            elif mode == 'strategy':
                store_key = f"{env_name}_strategy"
            else:
                store_key = env_name  # 'default'
            
            current_notes = self.sw_store.get(store_key) or ""
            obs_list = None
            if getattr(self, 'sw_track_frequency', False):
                obs_list = self._parse_numbered_observations(current_notes)
                current_notes = self._render_observations(obs_list) if obs_list else "(No observations yet)"
            elif not current_notes:
                current_notes = "(No notes yet)"
            prompt_template = _get_single_prompt_template(mode)
            
            # Version tracking must run before the strategy is regenerated.
            if explore and mode == 'strategy' and self.sw_store.get(store_key) and utilities:
                self._update_version_tracking(env_name, utilities)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_n = len(chunk)
                formatted_histories = ""
                for i, h in enumerate(chunk):
                    formatted_histories += f"=== Game {chunk_idx * chunk_size + i + 1} ===\n{h}\n\n"
                
                is_explore_call = explore and mode == 'strategy' and chunk_idx == len(chunks) - 1
                
                if is_explore_call:
                    from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_STRATEGY_EXPLORE
                    update_prompt = SW_UPDATE_PROMPT_STRATEGY_EXPLORE.format(
                        game_name=env_name,
                        game_intro=game_intro,
                        n=chunk_n,
                        old_strategy=current_notes,
                        game_histories=formatted_histories,
                        **self._explore_prompt_kwargs(env_name)
                    )
                else:
                    update_prompt = prompt_template.format(
                        game_name=env_name,
                        game_intro=game_intro,
                        n=chunk_n,
                        old_notes=current_notes,
                        game_histories=formatted_histories
                    )
                
                messages = [{"role": "user", "content": update_prompt}]
                raw_gen = ""
                try:
                    generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                    raw_gen = generations[0]
                    raw_notes = strip_thinking_block(raw_gen)
                    
                    if getattr(self, 'sw_track_frequency', False):
                        update_json = self._parse_freq_v2_json(raw_notes)
                        if update_json is not None:
                            obs_list = self._apply_freq_v2_update(obs_list, update_json)
                            current_notes = self._render_observations(obs_list)
                        else:
                            self.logger.warning(f"Failed to parse freq v2 json for chunk {chunk_idx}, ignoring chunk.")
                    else:
                        current_notes = raw_notes
                except Exception as e:
                    self.logger.error(f"Failed to generate SW update for chunk {chunk_idx}: {e}")
                    
                decision = None
                raw_audit = ""
                audit_prompt = ""
                if is_explore_call and raw_gen:
                    current_notes, decision, audit_prompt, raw_audit = self._apply_explore_decision(env_name, current_notes, "(No reputation in strategy-only mode)", formatted_histories)
                    
                self.logger.info(f'Chunk {chunk_idx} New Notes (N={chunk_n}):\\n{current_notes}')
                
                with open(log_file, 'a') as f:
                    f.write(f'=== CHUNK {chunk_idx} FLUSH (N={chunk_n}) SW UPDATE ===\\n')
                    f.write(f"PROMPT:\\n{update_prompt}\\n")
                    f.write(f'RAW THINKING:\\n{self._extract_thinking(raw_gen)}\\n')
                    f.write(f'SYNTHESIZED NOTES:\\n{current_notes}\\n')
                    f.write('=' * 50 + '\\n\\n')
                    if is_explore_call:
                        state = self._get_explore_state(env_name)
                        f.write('=== EXPLORE DECISION ===\\n')
                        f.write(f'DECISION: {decision or "SKIPPED (strategy call failed)"}\\n')
                        if audit_prompt:
                            f.write(f'AUDIT PROMPT:\\n{audit_prompt}\\n\\n')
                        if raw_audit:
                            f.write(f'AUDIT LOG:\\n{raw_audit}\\n\\n')
                        f.write(f'BATCH UTILITIES: {utilities}\\n')
                        f.write(f'CURRENT: total={state.get("current_total_utility", 0.0)}, '
                                f'games={state.get("current_games_played", 0)}\\n')
                        f.write(f'BEST: avg={state.get("best_avg_utility", None)}, '
                                f'games={state.get("best_games_played", 0)}\\n')
                        f.write('=' * 50 + '\\n\\n')
                    
            # Save final notes after all chunks are processed
            self.sw_store.update(store_key, current_notes)
            self.sw_store.save(self.sw_store_path)

    @staticmethod
    def _extract_thinking(raw: str) -> str:
        """Return the reasoning text inside the model's thinking block, for trace logging only."""
        if not raw:
            return "(empty generation)"
        for open_tag, close_tag in (("<think>", "</think>"), ("<thinking>", "</thinking>"), ("<thought>", "</thought>")):
            end = raw.rfind(close_tag)
            if end != -1:
                start = raw.find(open_tag)
                start = start + len(open_tag) if start != -1 and start < end else 0
                return raw[start:end].strip() or "(empty thinking block)"
        return "(no thinking block returned)"

    def flush_batch_updates_best_worst(self, winning_items: list, losing_items: list) -> None:
        """Perform SW update with winning and losing trajectories passed separately.
        
        Used by the 'best_worst' trajectory selection mode. The prompt separates
        winning and losing games and instructs the LLM to extract different types
        of observations from each set.
        
        Args:
            winning_items: list of (history, score) tuples for winning games
            losing_items: list of (history, score) tuples for losing games
        """
        if not winning_items and not losing_items:
            return
        
        n_win = len(winning_items)
        n_lose = len(losing_items)
        self.logger.info(f'{"-"*20}Batch SW Flush best_worst (W={n_win}, L={n_lose}){"-"*20}')
        
        env_name = self.current_game_name or "unknown"
        store_key = env_name
        current_notes = self.sw_store.get(store_key) or ""
        obs_list = None
        if getattr(self, 'sw_track_frequency', False):
            obs_list = self._parse_numbered_observations(current_notes)
            current_notes = self._render_observations(obs_list) if obs_list else "(No observations yet)"
        elif not current_notes:
            current_notes = "(No notes yet)"
        game_intro = getattr(self, 'current_game_intro', "Game rules unavailable.")
        log_file = self.sw_store_path.replace('.json', '_trace.log')
        
        from gamingbench.prompts.sliding_window_prompts import SW_UPDATE_PROMPT_BEST_WORST
        from gamingbench.utils.utils import strip_thinking_block
        
        template = SW_UPDATE_PROMPT_BEST_WORST
        if getattr(self, 'sw_track_frequency', False):
            from gamingbench.prompts.sliding_window_prompts import SW_TRACK_FREQUENCY_SUFFIX
            template = template + "\n" + SW_TRACK_FREQUENCY_SUFFIX
        
        # Format winning histories with scores
        formatted_winning = ""
        for i, (h, s) in enumerate(winning_items):
            formatted_winning += f"=== Winning Game {i + 1} (Score: {s}) ===\n{h}\n\n"
        if not winning_items:
            formatted_winning = "(No winning games in this batch)\n"
        
        # Format losing histories with scores
        formatted_losing = ""
        for i, (h, s) in enumerate(losing_items):
            formatted_losing += f"=== Losing Game {i + 1} (Score: {s}) ===\n{h}\n\n"
        if not losing_items:
            formatted_losing = "(No losing games in this batch)\n"
        
        update_prompt = template.format(
            game_name=env_name,
            game_intro=game_intro,
            old_notes=current_notes,
            n_win=n_win,
            n_lose=n_lose,
            winning_histories=formatted_winning,
            losing_histories=formatted_losing
        )
        
        messages = [{"role": "user", "content": update_prompt}]
        raw_gen = ""
        try:
            generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            raw_gen = generations[0]
            raw_notes = strip_thinking_block(raw_gen)
            
            if getattr(self, 'sw_track_frequency', False):
                update_json = self._parse_freq_v2_json(raw_notes)
                if update_json is not None:
                    obs_list = self._apply_freq_v2_update(obs_list, update_json)
                    current_notes = self._render_observations(obs_list)
                else:
                    self.logger.warning("Failed to parse freq v2 json, ignoring response.")
                    current_notes = self.sw_store.get(store_key) or ""
            else:
                current_notes = raw_notes
        except Exception as e:
            self.logger.error(f"Failed to generate best_worst SW update: {e}")
        
        self.logger.info(f'best_worst Notes (W={n_win}, L={n_lose}):\n{current_notes}')
        
        with open(log_file, 'a') as f:
            f.write(f'=== BEST_WORST FLUSH (W={n_win}, L={n_lose}) SW UPDATE ===\n')
            f.write(f"PROMPT:\n{update_prompt}\n")
            f.write(f'RAW THINKING:\n{self._extract_thinking(raw_gen)}\n')
            f.write(f'SYNTHESIZED NOTES:\n{current_notes}\n')
            f.write('=' * 50 + '\n\n')
        
        self.sw_store.update(store_key, current_notes)
        self.sw_store.save(self.sw_store_path)
