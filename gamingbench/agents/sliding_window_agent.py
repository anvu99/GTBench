import os
from typing import Tuple

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.sw.sw_store import SlidingWindowStore
from gamingbench.prompts.sliding_window_prompts import SW_INJECTION_PROMPT, SW_UPDATE_PROMPT


class SlidingWindowAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(SlidingWindowAgent, self).__init__(config, **kwargs)
        
        base_store_path = getattr(config, "sw_store_path", "sw_store.json")
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            name, ext = os.path.splitext(base_store_path)
            self.sw_store_path = f"{name}_{job_id}{ext}"
        else:
            self.sw_store_path = base_store_path
        
        self.sw_store = SlidingWindowStore()
        
        if os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)
            
        self.current_game_name = None
        
        # Batch mode variables
        self.batch_mode: bool = False
        self._last_batch_result = None  # Stores raw game_history string

    def set_storage_dir(self, storage_dir):
        """Called by main.py to align SW storage with the run's experiment folder."""
        self.sw_store_path = os.path.join(storage_dir, os.path.basename(self.sw_store_path))
        if os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking."""
        if not self.batch_mode and os.path.exists(self.sw_store_path):
            self.sw_store.load(self.sw_store_path)
            
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        self.game_count += 1
        
        # We don't care about opponent_key for SW, but we need the game name.
        # Since game_intro doesn't strictly have game_name isolated easily, 
        # we'll extract it from the env_name in step or post_game_update.

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        env_name = observations['env_name']
        self.current_game_name = env_name
        
        current_notes = self.sw_store.get(env_name)
        if current_notes:
            sw_injection = SW_INJECTION_PROMPT.format(
                game_name=env_name,
                notes_text=current_notes
            )
            from gamingbench.prompts.observation_prompts import construct_game_intro
            game_intro = construct_game_intro(env_name)
            observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + sw_injection, 1)
            
        return system_prompt, observation_prompt

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called at the end of the game."""
        self.current_game_name = env_name
        
        if self.batch_mode:
            self._last_batch_result = game_history
            self.logger.info('Batch mode: SlidingWindowAgent stored game history, synthesis deferred.')
            return
            
        # If not batch mode, just do it directly.
        self.flush_batch_updates([game_history])

    def flush_batch_updates(self, game_histories: list) -> None:
        """Perform a single unified SW update from a completed batch of N games."""
        if not game_histories:
            return
            
        n = len(game_histories)
        self.logger.info(f'-' * 20 + f'Batch SW Flush (N={n})' + '-' * 20)
        
        env_name = self.current_game_name or "unknown"
        current_notes = self.sw_store.get(env_name) or "(No notes yet)"
        
        # Split game_histories into chunks of at most 4 games to avoid 32k context limits
        chunk_size = 4
        chunks = [game_histories[i:i + chunk_size] for i in range(0, len(game_histories), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_n = len(chunk)
            formatted_histories = ""
            for i, h in enumerate(chunk):
                formatted_histories += f"=== Game {chunk_idx * chunk_size + i + 1} ===\n{h}\n\n"
                
            update_prompt = SW_UPDATE_PROMPT.format(
                game_name=env_name,
                game_intro=getattr(self, 'current_game_intro', "Game rules unavailable."),
                n=chunk_n,
                old_notes=current_notes,
                game_histories=formatted_histories
            )
            
            messages = [{"role": "user", "content": update_prompt}]
            try:
                from gamingbench.utils.utils import strip_thinking_block
                generations, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                current_notes = strip_thinking_block(generations[0])
            except Exception as e:
                self.logger.error(f"Failed to generate SW update for chunk {chunk_idx}: {e}")
                
            self.logger.info(f'Chunk {chunk_idx} New Notes (N={chunk_n}):\n{current_notes}')
            
            # Trace log
            log_file = self.sw_store_path.replace('.json', '_trace.log')
            with open(log_file, 'a') as f:
                f.write(f'=== CHUNK {chunk_idx} FLUSH (N={chunk_n}) SW UPDATE ===\n')
                f.write(f"PROMPT:\n{update_prompt}\n")
                f.write(f'SYNTHESIZED NOTES:\n{current_notes}\n')
                f.write('=' * 50 + '\n\n')
                
        # Save final notes after all chunks are processed
        self.sw_store.update(env_name, current_notes)
        self.sw_store.save(self.sw_store_path)
