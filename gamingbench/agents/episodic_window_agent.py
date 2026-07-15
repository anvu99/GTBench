import os
from typing import Tuple

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.sw.episodic_window_store import EpisodicWindowStore
from gamingbench.prompts.episodic_window_prompts import (
    EW_INJECTION_PROMPT,
    EW_ENDGAME_OBS_PROMPT,
    EW_WINDOW_SYNTHESIS_PROMPT
)


class EpisodicWindowAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(EpisodicWindowAgent, self).__init__(config, **kwargs)
        self.window_size = getattr(config, 'window_size', 10)
        self.ew_store_path = getattr(self, "ew_store_path", "ew_store.json")
        self.hive_mode = getattr(config, 'hive_mode', False)
        self.ew_store = EpisodicWindowStore()
        
        if os.path.exists(self.ew_store_path):
            self.ew_store.load(self.ew_store_path)
            
        self.current_game_name = None
        self.current_game_intro = None
        
        # Batch mode variables
        self.batch_mode: bool = False
        self._last_batch_result = None  # Stores raw game_history string

    def set_storage_dir(self, storage_dir):
        """Called by main.py to align EW storage with the run's experiment folder."""
        base = os.path.basename(self.ew_store_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
                
        self.ew_store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.ew_store_path):
            self.ew_store.load(self.ew_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking."""
        if not self.batch_mode and os.path.exists(self.ew_store_path):
            self.ew_store.load(self.ew_store_path)
            
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        self.game_count += 1
        
        # We don't care about opponent_key for EW, but we need the game name and intro
        self.current_game_intro = game_intro

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        obs_env = observations['env_name'].lower()
        # Use the specific game name (e.g. hanabi-small-custom) if it was set by the master thread
        env_name = getattr(self, 'current_game_name', obs_env)
        if not env_name or env_name == 'unknown':
            env_name = obs_env
            
        self.current_game_name = env_name
        
        current_notes = self.ew_store.get_notes(env_name)
        
        # Failsafe: In Hive mode, both agents share the exact same ew_store_path. 
        # If Player 1's in-memory notes are empty due to batch cloning skips, reload from disk.
        if not current_notes and getattr(self, 'hive_mode', False) and getattr(self, 'ew_store_path', None):
            if os.path.exists(self.ew_store_path):
                self.ew_store.load(self.ew_store_path)
                current_notes = self.ew_store.get_notes(env_name)
                
        if not current_notes and obs_env != env_name:
            current_notes = self.ew_store.get_notes(obs_env)
            
        ew_injection = ""
        if current_notes:
            ew_injection = EW_INJECTION_PROMPT.format(
                game_name=env_name,
                notes_text=current_notes
            )
            
        if getattr(self, 'hive_mode', False):
            from gamingbench.prompts.hive_prompts import HIVE_MEMORY_NOTICE
            if ew_injection:
                ew_injection = HIVE_MEMORY_NOTICE + "\n\n" + ew_injection
            else:
                ew_injection = HIVE_MEMORY_NOTICE
                
        if ew_injection:
            observation_prompt = ew_injection + "\n\n" + observation_prompt
            
        return system_prompt, observation_prompt

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called at the end of the game."""
        self.current_game_name = env_name.lower()
        
        if self.batch_mode:
            self._last_batch_result = game_history
            self.logger.info('Batch mode: EpisodicWindowAgent stored game history, synthesis deferred.')
            return
            
        # If not batch mode, just do it directly.
        self.flush_batch_updates([game_history])

    def flush_batch_updates(self, game_histories: list) -> None:
        """Perform a single unified EW update from a completed batch of N games."""
        if not game_histories:
            return
            
        n = len(game_histories)
        self.logger.info(f'-' * 20 + f'Batch EW Flush (N={n})' + '-' * 20)
        
        env_name = (self.current_game_name or "unknown").lower()
        game_intro = getattr(self, 'current_game_intro', "Game rules unavailable.")
        if not game_intro or game_intro == "Game rules unavailable.":
            from gamingbench.prompts.observation_prompts import construct_game_intro
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
            
        if getattr(self, 'hive_mode', False):
            from gamingbench.prompts.hive_prompts import HIVE_UPDATE_NOTICE
            game_intro = HIVE_UPDATE_NOTICE + "\n\n" + game_intro
            
        from gamingbench.utils.utils import strip_thinking_block
        
        # 1. Process each game history to generate a short observation
        for i, game_history in enumerate(game_histories):
            obs_prompt = EW_ENDGAME_OBS_PROMPT.format(
                game_name=env_name,
                game_intro=game_intro,
                game_history=game_history
            )
            messages = [{"role": "user", "content": obs_prompt}]
            try:
                thinking_enabled = getattr(self.model, 'enable_thinking', False)
                retries = 0
                while True:
                    generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                    raw_gen = generations[0]
                    has_tag = any(tag in raw_gen for tag in ["<think>", "</think>", "<thought>", "</thought>"])
                    if not thinking_enabled or has_tag or retries >= 2:
                        break
                    retries += 1
                    self.logger.warning(f"Missing thinking tag in EW endgame observation, retrying ({retries}/2)...")
                    
                if thinking_enabled and not has_tag:
                    self.logger.error("Failed to generate thinking tags for EW observation after retries.")
                    observation = "Game observation generation failed."
                else:
                    observation = strip_thinking_block(raw_gen).strip()
                    
                self.ew_store.add_observation(env_name, observation, self.window_size)
                self.logger.info(f'Game {i+1}/{n} Observation:\n{observation}')
            except Exception as e:
                self.logger.error(f"Failed to generate EW observation for game {i+1}: {e}")
                
        # 2. Synthesize all observations in the window into new notes
        observations_list = self.ew_store.get_observations(env_name)
        if observations_list:
            formatted_observations = ""
            for i, obs in enumerate(observations_list):
                # Reverse order so the most recent is clearly marked? Or just chronological.
                # Chronological: 1 is oldest in window, len is newest.
                formatted_observations += f"=== Game Observation {i + 1} ===\n{obs}\n\n"
                
            synthesis_prompt = EW_WINDOW_SYNTHESIS_PROMPT.format(
                game_name=env_name,
                game_intro=game_intro,
                n=len(observations_list),
                observations=formatted_observations
            )
            messages = [{"role": "user", "content": synthesis_prompt}]
            try:
                thinking_enabled = getattr(self.model, 'enable_thinking', False)
                retries = 0
                while True:
                    generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                    raw_gen = generations[0]
                    has_tag = any(tag in raw_gen for tag in ["<think>", "</think>", "<thought>", "</thought>"])
                    if not thinking_enabled or has_tag or retries >= 2:
                        break
                    retries += 1
                    self.logger.warning(f"Missing thinking tag in EW synthesis, retrying ({retries}/2)...")
                    
                if thinking_enabled and not has_tag:
                    self.logger.error("Failed to generate thinking tags for EW synthesis after retries.")
                    new_notes = self.ew_store.get_notes(env_name) or "Failed to synthesize notes."
                else:
                    new_notes = strip_thinking_block(raw_gen).strip()
                    
                self.ew_store.update_notes(env_name, new_notes)
                self.logger.info(f'Synthesized EW Notes from {len(observations_list)} observations:\n{new_notes}')
            except Exception as e:
                self.logger.error(f"Failed to generate EW synthesis: {e}")
                new_notes = "Failed to synthesize notes."
                
            # Trace log
            log_file = self.ew_store_path.replace('.json', '_trace.log')
            with open(log_file, 'a') as f:
                f.write(f'=== BATCH FLUSH (N={n} games, Window={len(observations_list)} obs) EW UPDATE ===\n')
                f.write(f"OBSERVATIONS COLLECTED:\n{formatted_observations}\n")
                f.write(f"SYNTHESIS PROMPT:\n{synthesis_prompt}\n")
                f.write(f'SYNTHESIZED NOTES:\n{new_notes}\n')
                f.write('=' * 50 + '\n\n')
                
            # Save final notes
            self.ew_store.save(self.ew_store_path)
