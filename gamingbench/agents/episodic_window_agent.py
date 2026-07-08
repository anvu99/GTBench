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
        
        base_store_path = getattr(config, "ew_store_path", "ew_store.json")
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            name, ext = os.path.splitext(base_store_path)
            self.ew_store_path = f"{name}_{job_id}{ext}"
        else:
            self.ew_store_path = base_store_path
        
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
        self.ew_store_path = os.path.join(storage_dir, os.path.basename(self.ew_store_path))
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
        
        env_name = observations['env_name']
        self.current_game_name = env_name
        
        current_notes = self.ew_store.get_notes(env_name)
        if current_notes:
            ew_injection = EW_INJECTION_PROMPT.format(
                game_name=env_name,
                notes_text=current_notes
            )
            from gamingbench.prompts.observation_prompts import construct_game_intro
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False))
            observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + ew_injection, 1)
            
        return system_prompt, observation_prompt

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called at the end of the game."""
        self.current_game_name = env_name
        
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
        
        env_name = self.current_game_name or "unknown"
        game_intro = getattr(self, 'current_game_intro', "Game rules unavailable.")
        if not game_intro or game_intro == "Game rules unavailable.":
            from gamingbench.prompts.observation_prompts import construct_game_intro
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False))
            
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
                generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                observation = strip_thinking_block(generations[0]).strip()
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
                generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                new_notes = strip_thinking_block(generations[0]).strip()
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
