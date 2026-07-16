import os
import re
from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.ltm_store import LTMStore, OpponentLTMStore
from gamingbench.ltm.prompts import (
    LTM_INJECTION_PROMPT, SELF_LTM_INJECTION_PROMPT, PROACTIVE_LTM_INJECTION_PROMPT,
    WINDOW_SUMMARIZE_PROMPT, GRADIENT_ENGINE_PROMPT, TGD_SYNTHESIS_PROMPT
)
from gamingbench.ltm.gradient_engine import run_gradient_engine
from gamingbench.ltm.tgd_synthesizer import run_tgd_synthesis
from gamingbench.ltm.self_gradient_engine import run_self_gradient_engine
from gamingbench.ltm.self_tgd_synthesizer import run_self_tgd_synthesis
from gamingbench.ltm.proactive_gradient_engine import run_proactive_gradient_engine
from gamingbench.ltm.proactive_tgd_synthesizer import run_proactive_tgd_synthesis
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.prompts.observation_prompts import construct_observation_prompt
import threading

_trace_log_lock = threading.Lock()






class LTMAgent(PromptAgent):

    def __init__(self, config, **kwargs):
        super(LTMAgent, self).__init__(config, **kwargs)
        
        self.summarize_every = getattr(config, "summarize_every", 5)
        
        base_store_path = getattr(config, "ltm_store_path", "ltm_store.json")
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            name, ext = os.path.splitext(base_store_path)
            self.ltm_store_path = f"{name}_{job_id}{ext}"
        else:
            self.ltm_store_path = base_store_path
        self.ltm_store = LTMStore()
        
        if os.path.exists(self.ltm_store_path):
            self.ltm_store.load(self.ltm_store_path)

        # ── Self-LTM store ────────────────────────────────────────────────────
        self.self_ltm_store_path = self.ltm_store_path.replace('ltm_store', 'self_ltm_store')
        self.self_ltm_store = LTMStore()
        
        self.hive_mode = getattr(config, "hive_mode", False)
        if os.path.exists(self.self_ltm_store_path):
            self.self_ltm_store.load(self.self_ltm_store_path)
            
        # ── Proactive LTM store ───────────────────────────────────────────────
        self.proactive_ltm_store_path = self.ltm_store_path.replace('ltm_store', 'overall_strategy')
        self.proactive_ltm_store = LTMStore()
        
        if os.path.exists(self.proactive_ltm_store_path):
            self.proactive_ltm_store.load(self.proactive_ltm_store_path)
            
        # ── Partner LTM store ────────────────────────────────────────────────
        self.partner_ltm_store = LTMStore()
        self.partner_ltm_store_path = None
        self.partner_opponent_key = None
            
        self.window_summaries = []
        self.recent_internal_reasoning = []
        self.move_count = 0
        self.current_opponent_key = None
        self.current_game_intro = None
        # ── Batch mode ───────────────────────────────────────────────────────
        # When batch_mode=True, post_game_update() stores gradient data and
        # returns without synthesizing. The runner calls flush_batch_updates()
        # after all N games in the batch complete.
        self.batch_mode: bool = False
        self._last_batch_result = None  # (structural_report, game_scores) | None

    def set_storage_dir(self, storage_dir):
        """Called by main.py to align LTM storage with the run's experiment folder."""
        base = os.path.basename(self.ltm_store_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate' or getattr(self, 'hive_mode', False):
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
                
        self.ltm_store_path = os.path.join(storage_dir, base)
        self.self_ltm_store_path = self.ltm_store_path.replace('ltm_store', 'self_ltm_store')
        self.proactive_ltm_store_path = self.ltm_store_path.replace('ltm_store', 'overall_strategy')
        # Reload if it happens to already exist in this specific directory
        if os.path.exists(self.ltm_store_path):
            self.ltm_store.load(self.ltm_store_path)
        if os.path.exists(self.self_ltm_store_path):
            self.self_ltm_store.load(self.self_ltm_store_path)
        if os.path.exists(self.proactive_ltm_store_path):
            self.proactive_ltm_store.load(self.proactive_ltm_store_path)

    def set_partner_store(self, path: str, my_key: str):
        """Called by main.py in Hive mode to inject the partner's LTM store."""
        self.partner_ltm_store_path = path
        self.partner_opponent_key = my_key
        if path and os.path.exists(path):
            self.partner_ltm_store.load(path)

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking."""
        # In batch mode the LTM snapshot was pre-loaded in-memory by clone_agent_for_batch().
        # Do NOT reload from disk — the clone's ltm_store_path is intentionally '/dev/null'
        # (non-JSON) and reloading would raise JSONDecodeError.
        if not self.batch_mode and os.path.exists(self.ltm_store_path):
            self.ltm_store.load(self.ltm_store_path)
        if not self.batch_mode and os.path.exists(self.self_ltm_store_path):
            self.self_ltm_store.load(self.self_ltm_store_path)
        if not self.batch_mode and os.path.exists(self.proactive_ltm_store_path):
            self.proactive_ltm_store.load(self.proactive_ltm_store_path)
        if not self.batch_mode and getattr(self, 'hive_mode', False) and self.partner_ltm_store_path:
            if os.path.exists(self.partner_ltm_store_path):
                self.partner_ltm_store.load(self.partner_ltm_store_path)
            
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        self.game_count += 1
        self.window_summaries = []
        self.recent_internal_reasoning = []
        self.move_count = 0
        self.current_game_intro = game_intro
        
        if getattr(self, 'hive_mode', False) and not isinstance(opponent_key, list):
            # Force separate memory mode for Hive
            opponent_key = opponent_key.split('+')
            
        if isinstance(opponent_key, list):
            self.current_opponent_keys = opponent_key
            self.memory_mode = 'separate'
            self.current_opponent_key = opponent_key[0] if len(opponent_key) == 1 else None
        else:
            self.current_opponent_keys = [opponent_key]
            self.memory_mode = 'combined'
            self.current_opponent_key = opponent_key

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        current_ltm = None
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            ltms = []
            for key in self.current_opponent_keys:
                ltm = self.ltm_store.get(key)
                if ltm and ltm.strip() != "(No signals currently stored)":
                    peer_id = key.split(':')[0] if ':' in key else key
                    active_ltm = ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
                    # Replaced `f"TEAMMATE {peer_id}"` with `peer_id` so the prompt refers directly to "Player X" instead of "TEAMMATE LTMCotAgent"
                    ltms.append(LTM_INJECTION_PROMPT.format(opponent_id=peer_id, ltm_text=active_ltm))
            if ltms:
                current_ltm = "\n\n".join(ltms)
        else:
            if self.current_opponent_key:
                current_ltm = self.ltm_store.get(self.current_opponent_key)
                
            # Failsafe: In Hive mode, both agents share the exact same ltm_store_path. 
            # If Player 1's in-memory memory is empty due to batch cloning skips, reload from disk.
            if not current_ltm and getattr(self, 'hive_mode', False) and getattr(self, 'ltm_store_path', None):
                if os.path.exists(self.ltm_store_path):
                    self.ltm_store.load(self.ltm_store_path)
                    if self.current_opponent_key:
                        current_ltm = self.ltm_store.get(self.current_opponent_key)
                        
            if current_ltm and current_ltm.strip() != "(No signals currently stored)":
                active_ltm = current_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
                # Extracted peer_id from current_opponent_key to use "Player X" instead of the hardcoded string "the opponent"
                peer_id = self.current_opponent_key.split(':')[0] if ':' in self.current_opponent_key else self.current_opponent_key
                current_ltm = LTM_INJECTION_PROMPT.format(opponent_id=peer_id, ltm_text=active_ltm)
                
        if getattr(self, 'hive_mode', False):
            from gamingbench.prompts.hive_prompts import HIVE_MEMORY_NOTICE
            if current_ltm:
                current_ltm = HIVE_MEMORY_NOTICE + "\n\n" + current_ltm
            else:
                current_ltm = HIVE_MEMORY_NOTICE
                
        if current_ltm:
            observation_prompt = current_ltm + "\n\n" + observation_prompt

        current_proactive_ltm = self.proactive_ltm_store.get("__overall__")
        if current_proactive_ltm and current_proactive_ltm.strip() != "(No signals currently stored)":
            active_proactive_ltm = current_proactive_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
            proactive_ltm_injection = PROACTIVE_LTM_INJECTION_PROMPT.format(
                proactive_ltm_text=active_proactive_ltm
            )
            # Inject proactive LTM right before the opponent LTM
            if current_ltm:
                observation_prompt = observation_prompt.replace(current_ltm, proactive_ltm_injection + "\n\n" + current_ltm, 1)
            else:
                from gamingbench.prompts.observation_prompts import construct_game_intro
                env_name = observations['env_name']
                game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
                observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + proactive_ltm_injection, 1)

        current_self_ltm = self.self_ltm_store.get("__self__")
        if current_self_ltm and current_self_ltm.strip() != "(No signals currently stored)":
            active_self_ltm = current_self_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
            self_ltm_injection = SELF_LTM_INJECTION_PROMPT.format(
                self_ltm_text=active_self_ltm
            )
            from gamingbench.prompts.observation_prompts import construct_game_intro
            env_name = observations['env_name']
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
            # Inject self-LTM after the opponent LTM (or after game intro if no opponent LTM)
            if current_ltm:
                inject_after = current_ltm
            else:
                inject_after = game_intro
            observation_prompt = observation_prompt.replace(inject_after, inject_after + "\n\n" + self_ltm_injection, 1)
            
        if getattr(self, 'hive_mode', False) and self.partner_opponent_key:
            partner_view = self.partner_ltm_store.get(self.partner_opponent_key)
            if partner_view and partner_view.strip() != "(No signals currently stored)":
                active_partner_view = partner_view.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
                from gamingbench.prompts.hive_prompts import PARTNER_VIEW_OF_ME_PROMPT
                partner_injection = PARTNER_VIEW_OF_ME_PROMPT.format(partner_view_text=active_partner_view)
                if 'self_ltm_injection' in locals() and self_ltm_injection in observation_prompt:
                    observation_prompt = observation_prompt.replace(self_ltm_injection, self_ltm_injection + "\n\n" + partner_injection, 1)
                elif current_ltm:
                    observation_prompt = observation_prompt.replace(current_ltm, current_ltm + "\n\n" + partner_injection, 1)
                else:
                    if 'game_intro' not in locals():
                        from gamingbench.prompts.observation_prompts import construct_game_intro
                        game_intro = construct_game_intro(observations['env_name'], enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
                    observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + partner_injection, 1)
            
        return system_prompt, observation_prompt

    def step(self, observations):
        """
        Runs the full LTM-integrated step, including optional summarization.
        """
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []
        
        # 1. Summarization check (before action)
        if self.move_count > 0 and self.move_count % self.summarize_every == 0:
            sum_query = self._run_window_summarization(observations)
            if sum_query:
                query_list.append(sum_query)

        # 2. Prepare action prompt
        system_prompt, observation_prompt = self._build_prompts(observations)

        # Borrow the action regex from the base step prompt constructor.
        step_instruct = self.step_prompt_constructor(observations)
        regex = step_instruct['regex']
        action_format = step_instruct.get('format', observations.get('legal_moves', []))

        # Build action reminder (same logic as base CoT step prompt).
        legal_moves = observations.get('legal_moves', [])
        if len(legal_moves) <= 10:
            action_reminder = f"Remember, you can only choose one move from the legal actions which is {legal_moves}"
        else:
            action_reminder = "Remember, you can only choose one move from the legal actions."

        # Check which databases are loaded for this turn.
        has_opponent_ltm = bool(
            self.current_opponent_key and self.ltm_store.get(self.current_opponent_key)
        )
        has_self_ltm = bool(self.self_ltm_store.get("__self__"))
        has_proactive_ltm = bool(self.proactive_ltm_store.get("__overall__"))

        if has_opponent_ltm or has_self_ltm or has_proactive_ltm:
            # Structured Thought format — signal evaluation is woven into
            # the reasoning itself so the agent cannot skip it or apply it post-hoc.
            from gamingbench.prompts.regex_and_format import get_step_env_regex_and_format
            _, fmt = get_step_env_regex_and_format(observations.get('env_name', ''), turn_type=observations.get('turn_type'))

            scan_sources = []
            if has_proactive_ltm:
                scan_sources.append("OVERALL STRATEGY DATABASE")
            if has_self_ltm:
                scan_sources.append("SELF-REPUTATION DATABASE")
            if has_opponent_ltm:
                scan_sources.append("OPPONENT REPUTATION DATABASE")
            scan_label = " and ".join(scan_sources)

            cot_stages = []
            cot_stages.append("[Board Analysis] First, carefully parse the board state. Identify where your pieces are, where the opponent's pieces are, and which direction you are moving.")
            
            if has_proactive_ltm:
                cot_stages.append("[Proactive Strategy] Review the OVERALL STRATEGY DATABASE. Identify any proactive strategies or traps you wish to deploy right now, and extract their Policies.")
            
            if has_opponent_ltm or has_self_ltm:
                scan_str = "OPPONENT REPUTATION DATABASE and SELF-REPUTATION DATABASE" if (has_opponent_ltm and has_self_ltm) else ("OPPONENT REPUTATION DATABASE" if has_opponent_ltm else "SELF-REPUTATION DATABASE")
                cot_stages.append(f"[Signal Scan] For each signal in your {scan_str}, carefully reason through whether its 'When' condition is met by the current board state and game context. Conclude clearly whether it fires.")
            
            if has_opponent_ltm:
                cot_stages.append("[Opponent Policy Synthesis] Synthesize the active Policies from any firing OPPONENT signals into a coherent reactive strategy for this move.")
            
            cot_stages.append("[Candidate Move] Formulate your top candidate move based on the board state, prioritizing reactive strategies from the opponent database (if any fire) first, and then executing proactive strategies (if any) second.")
                
            if has_self_ltm:
                cot_stages.append("[Guardrail Verification] Check if your candidate move matches the 'What' field of any firing SELF signals. If so, execute their 'Verification' calculation to ensure the 'Risk' will not materialize. If the verification shows the risk will occur, you MUST reject the candidate and formulate a new move.")
                
            cot_stages.append("[Final Decision] State your final chosen move based on the reasoning above.")
            
            stages_text = "\n\n".join(cot_stages)

            step_prompt = f"""As you reason through your move, ensure you internally process the following stages:

{stages_text}

Your output must be in the following format strictly:

Action:
Your action wrapped by <>, i.e., {fmt}

{action_reminder}
"""
        else:
            # No databases yet — use the standard CoT step prompt unchanged.
            step_prompt = step_instruct['prompt']

        observation_prompt = observation_prompt + '\n' + step_prompt
        
        msgs = self.construct_init_messages(system_prompt, observation_prompt)

        responses, query = self.llm_query(
            msgs, n=self.num_generations, stop=None, prompt_type='move')
        query_list.append(query)
        
        self.recent_internal_reasoning.append(f"Move {self.move_count} Reasoning:\n{responses[0]}")

        self.logger.info(f'Prompt: {observation_prompt}')
        self.logger.info(f'Response: {responses}')

        moves = self.parse_with_regex(responses, regex)
        if len(moves) != 0:
            move = self.post_processing(moves, majority_vote=self.voting)
        else:
            move = ""

        # Log tracing information
        log_file = getattr(self, '_parent_store_path', self.ltm_store_path).replace('.json', '_trace.log')
        with _trace_log_lock:
            with open(log_file, "a") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} MOVE {self.move_count} STEP ===\n")
                if getattr(self, 'game_count', 0) == 1:
                    f.write(f"SYSTEM PROMPT:\n{system_prompt}\n")
                    f.write(f"OBSERVATION PROMPT:\n{observation_prompt}\n")
                f.write(f"RESPONSE:\n{responses}\n")
                f.write("=" * 50 + "\n\n")

        self.move_count += 1
        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        has_opponent_ltm = bool(self.current_opponent_key and self.ltm_store.get(self.current_opponent_key))
        has_self_ltm = bool(self.self_ltm_store.get("__self__"))
        has_proactive_ltm = bool(self.proactive_ltm_store.get("__overall__"))
        
        if not (has_opponent_ltm or has_self_ltm or has_proactive_ltm):
            return super().chat_step(observations, chat_history_str)
            
        self.logger.info('-' * 20 + f'{self.agent_name} Chat Generation (LTM)' + '-' * 20)
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = self._build_prompts(observations)
        
        cot_stages = []
        cot_stages.append("[Context Analysis] Briefly analyze the current game state and the ongoing chat history. Identify the current strategic situation.")
        
        if has_proactive_ltm:
            cot_stages.append("[Proactive Chat Strategy] Review the OVERALL STRATEGY DATABASE for any strategies of Type: 'Chat'. Decide if you should deploy one now to misdirect, bluff, or manipulate the opponent.")
            
        if has_opponent_ltm or has_self_ltm:
            scan_str = "OPPONENT REPUTATION DATABASE and SELF-REPUTATION DATABASE" if (has_opponent_ltm and has_self_ltm) else ("OPPONENT REPUTATION DATABASE" if has_opponent_ltm else "SELF-REPUTATION DATABASE")
            cot_stages.append(f"[Signal Scan] Check your {scan_str} to see if any signals fire based on the current situation, and how they should influence your communication.")
            
        cot_stages.append("[Message Formulation] Formulate your final chat message based on the reasoning above. If no strategic message is needed, you may remain silent or formulate a generic message.")
        
        stages_text = "\n\n".join(cot_stages)
        
        if observations.get('env_name') == 'cooperative_negotiation':
            from gamingbench.prompts.chat_prompts import COOP_CHAT_INSTRUCTION
            instruction = COOP_CHAT_INSTRUCTION
        else:
            from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION
            instruction = CHAT_INSTRUCTION

        step_prompt = f"""As you reason through your chat message, ensure you internally process the following stages:

{stages_text}

Then, following this instruction:
{instruction}
"""
        observation_prompt = observation_prompt + '\n\n' + step_prompt
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        try:
            from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            message = strip_thinking_block(responses[0]).strip()
            message = strip_chat_tags(message)
            self.logger.info(f"Chat Generated: {message}")
            
            # Log tracing information
            log_file = getattr(self, '_parent_store_path', getattr(self, 'ltm_store_path', 'default_trace.log')).replace('.json', '_trace.log')
            try:
                with _trace_log_lock:
                    with open(log_file, "a") as f:
                        f.write(f"=== GAME {getattr(self, 'game_count', 0)} MOVE {self.move_count} CHAT ===\n")
                        if getattr(self, 'game_count', 0) == 1:
                            f.write(f"SYSTEM PROMPT:\n{system_prompt}\n")
                            f.write(f"OBSERVATION PROMPT:\n{observation_prompt}\n")
                        f.write(f"RESPONSE:\n{responses}\n")
                        f.write("=" * 50 + "\n\n")
            except Exception as log_e:
                self.logger.error(f"Failed to write chat trace log: {log_e}")
                
            return message, query
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return "", None

    def _run_window_summarization(self, observations):
        """Fires a separate standalone LLM call to summarize the recent window."""
        self.logger.info('-' * 20 + f'{self.agent_name} Summarization' + '-' * 20)
        current_ltm = None
        if getattr(self, 'memory_mode', 'combined') == 'separate':
            ltms = []
            for key in self.current_opponent_keys:
                ltm = self.ltm_store.get(key)
                if ltm:
                    peer_id = key.split(':')[0] if ':' in key else key
                    active_ltm = ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
                    # Replaced `f"TEAMMATE {peer_id}"` with `peer_id` to maintain consistent "Player X" naming in summarization
                    ltms.append(LTM_INJECTION_PROMPT.format(opponent_id=peer_id, ltm_text=active_ltm))
            if ltms:
                current_ltm = "\n\n".join(ltms)
        else:
            if self.current_opponent_key:
                current_ltm = self.ltm_store.get(self.current_opponent_key)
            if current_ltm:
                active_ltm = current_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
                # Extracted peer_id to replace "the opponent" with "Player X" in the summarization prompt
                peer_id = self.current_opponent_key.split(':')[0] if ':' in self.current_opponent_key else self.current_opponent_key
                current_ltm = LTM_INJECTION_PROMPT.format(opponent_id=peer_id, ltm_text=active_ltm)
                
        env_name = observations.get('env_name', 'unknown')
        chat_context = observations.get('chat_context', '')
        
        sys_content = construct_system_prompt(env_name) if env_name != 'unknown' else "You are a powerful gaming agent who can make proper decisions to achieve your objective (either defeating the opponent or coordinating successfully with your partner) in gaming tasks. You are a helpful assistant that strictly follows the user's instructions. You must answer your questions by choosing one of the legal moves given by the user!"
        
        game_intro = self.current_game_intro or "Game rules unavailable."
        user_prompt_parts = [game_intro]
        
        if current_ltm:
            user_prompt_parts.append(current_ltm)

        current_self_ltm = self.self_ltm_store.get("__self__")
        if current_self_ltm:
            active_self_ltm = current_self_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
            self_ltm_injection = SELF_LTM_INJECTION_PROMPT.format(
                self_ltm_text=active_self_ltm
            )
            user_prompt_parts.append(self_ltm_injection)
            
        if getattr(self, 'enable_chat', False):
            user_prompt_parts.append("In this game version, players are allowed to communicate with each other. However, the chat channel is NOT a set of binding rules. It is simply a transcript of player dialogue. Do NOT treat the chat as hardcoded rules you must follow. Your ultimate goal is to win the game, and you should evaluate the chat strategically.")
            if chat_context and chat_context != "No messages yet.":
                injection = f"--- ONGOING CHAT ---\n{chat_context}"
                user_prompt_parts.append(injection)
                
        board_state = construct_observation_prompt(observations, env_name)
        user_prompt_parts.append(f"--- CURRENT GAME STATE ---\n{board_state}")
        
        if getattr(self, 'recent_internal_reasoning', []):
            user_prompt_parts.append("--- YOUR RECENT INTERNAL REASONING ---\n" + "\n\n".join(self.recent_internal_reasoning))
            self.recent_internal_reasoning.clear()
            
        user_prompt_parts.append(WINDOW_SUMMARIZE_PROMPT.format(K=self.summarize_every))
        
        user_content = "\n\n".join(user_prompt_parts)
        
        messages = [
            {'role': 'system', 'content': sys_content},
            {'role': 'user', 'content': user_content}
        ]
        
        try:
            from gamingbench.utils.utils import strip_thinking_block
            
            thinking_enabled = getattr(self.model, 'enable_thinking', False)
            retries = 0
            while True:
                generations, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                raw_gen = generations[0]
                has_tag = any(tag in raw_gen for tag in ["<think>", "</think>", "<thought>", "</thought>"])
                if not thinking_enabled or has_tag or retries >= 2:
                    break
                retries += 1
                self.logger.warning(f"Missing thinking tag in window summarization, retrying ({retries}/2)...")
                
            if thinking_enabled and not has_tag:
                self.logger.error("Failed to generate thinking tags for summarization after retries.")
                summary = "Game/Opponent summary: [Summary failed]\n\nReasoning memory: [Reasoning failed]"
            else:
                summary = strip_thinking_block(raw_gen)
                
            self.window_summaries.append(f"Window ending at move {self.move_count}:\n{summary}")
            self.logger.info(f'Summarization: {summary}')
            
            # Log tracing information
            log_file = getattr(self, '_parent_store_path', self.ltm_store_path).replace('.json', '_trace.log')
            with _trace_log_lock:
                with open(log_file, "a") as f:
                    f.write(f"=== GAME {getattr(self, 'game_count', 0)} MOVE {self.move_count} SUMMARIZATION ===\n")
                    f.write(f"SYSTEM PROMPT:\n{sys_content}\n")
                    f.write(f"USER PROMPT:\n{user_content}\n")
                    f.write(f"RESPONSE:\n{summary}\n")
                    f.write("=" * 50 + "\n\n")

            return query
        except Exception as e:
            self.logger.error(f"[LTMAgent] Summarization failed: {e}")
            return None

    def _run_final_summarization(self, game_history: str, final_board_state: str, env_name: str = 'unknown'):
        """Runs a final summarization at the end of the game to capture the last few moves."""
        self.logger.info('-' * 20 + f'{self.agent_name} Final Summarization' + '-' * 20)
        current_ltm = self.ltm_store.get(self.current_opponent_key) if self.current_opponent_key else None
        
        sys_content = construct_system_prompt(env_name) if env_name != 'unknown' else "You are a powerful gaming agent who can make proper decisions to achieve your objective (either defeating the opponent or coordinating successfully with your partner) in gaming tasks. You are a helpful assistant that strictly follows the user's instructions. You must answer your questions by choosing one of the legal moves given by the user!"
        
        game_intro = self.current_game_intro or "Game rules unavailable."
        user_prompt_parts = [game_intro]
        
        if current_ltm:
            active_ltm = current_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
            # Extracted peer_id to replace "the opponent" with "Player X" in the final summarization prompt
            peer_id = self.current_opponent_key.split(':')[0] if ':' in self.current_opponent_key else self.current_opponent_key
            ltm_injection = LTM_INJECTION_PROMPT.format(
                opponent_id=peer_id,
                ltm_text=active_ltm
            )
            user_prompt_parts.append(ltm_injection)

        current_self_ltm = self.self_ltm_store.get("__self__")
        if current_self_ltm:
            active_self_ltm = current_self_ltm.split("--- GRAVEYARD OF FAILED STRATEGIES ---")[0].strip()
            self_ltm_injection = SELF_LTM_INJECTION_PROMPT.format(
                self_ltm_text=active_self_ltm
            )
            user_prompt_parts.append(self_ltm_injection)
            
        user_prompt_parts.append(f"--- FULL GAME HISTORY ---\n{game_history}\n\n--- FINAL BOARD STATE ---\n{final_board_state}")
        
        if getattr(self, 'recent_internal_reasoning', []):
            user_prompt_parts.append("--- YOUR RECENT INTERNAL REASONING ---\n" + "\n\n".join(self.recent_internal_reasoning))
            self.recent_internal_reasoning.clear()
            
        user_prompt_parts.append(WINDOW_SUMMARIZE_PROMPT.format(K="final"))
            
        user_content = "\n\n".join(user_prompt_parts)
            
        messages = [
            {'role': 'system', 'content': sys_content},
            {'role': 'user', 'content': user_content}
        ]
        
        try:
            from gamingbench.utils.utils import strip_thinking_block
            
            thinking_enabled = getattr(self.model, 'enable_thinking', False)
            retries = 0
            while True:
                generations, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
                raw_gen = generations[0]
                has_tag = any(tag in raw_gen for tag in ["<think>", "</think>", "<thought>", "</thought>"])
                if not thinking_enabled or has_tag or retries >= 2:
                    break
                retries += 1
                self.logger.warning(f"Missing thinking tag in final summarization, retrying ({retries}/2)...")
                
            if thinking_enabled and not has_tag:
                self.logger.error("Failed to generate thinking tags for final summarization after retries.")
                summary = "Game/Opponent summary: [Summary failed]\n\nReasoning memory: [Reasoning failed]"
            else:
                summary = strip_thinking_block(raw_gen)
                
            self.window_summaries.append(f"Final Window ending at game over:\n{summary}")
            self.logger.info(f'Final Summarization: {summary}')
            
            # Log tracing information
            log_file = getattr(self, '_parent_store_path', self.ltm_store_path).replace('.json', '_trace.log')
            with _trace_log_lock:
                with open(log_file, "a") as f:
                    f.write(f"=== GAME {getattr(self, 'game_count', 0)} FINAL SUMMARIZATION ===\n")
                    f.write(f"SYSTEM PROMPT:\n{sys_content}\n")
                    f.write(f"USER PROMPT:\n{user_content}\n")
                    f.write(f"RESPONSE:\n{summary}\n")
                    f.write("=" * 50 + "\n\n")

            return query
        except Exception as e:
            self.logger.error(f"[LTMAgent] Final summarization failed: {e}")
            return None

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Runs the LTM gradient and synthesis pipelines after a game."""
        player_index = getattr(self, 'current_player_index', None)
        if player_index is not None:
            game_history = game_history.replace(f"Player {player_index}", "You")

        if not getattr(self, 'current_opponent_keys', None):
            return

        game_intro_for_update = self.current_game_intro or "Game rules unavailable."

        if self.move_count > len(self.window_summaries) * self.summarize_every:
            self._run_final_summarization(game_history, final_board_state, env_name)

        self.logger.info('-' * 20 + f'{self.agent_name} Post-Game LTM Update' + '-' * 20)
        window_summaries_str = "\n\n".join(self.window_summaries) if self.window_summaries else "No window summaries recorded."

        from concurrent.futures import ThreadPoolExecutor
        from gamingbench.prompts.observation_prompts import construct_game_history_legend
        try:
            game_history_legend = construct_game_history_legend(env_name)
        except Exception:
            game_history_legend = "Game history legend unavailable."
            
        opp_results = {}
        with ThreadPoolExecutor(max_workers=max(2, len(self.current_opponent_keys)+1)) as ex:
            if getattr(self, 'memory_mode', 'combined') == 'separate':
                from gamingbench.ltm.gradient_engine import run_separate_gradient_engine
                future_opps = {}
                for key in self.current_opponent_keys:
                    peer_id = key.split(':')[0] if ':' in key else key
                    current_ltm = self.ltm_store.get(key) or "(No memory yet)"
                    future_opps[key] = ex.submit(
                        run_separate_gradient_engine,
                        model=self.model,
                        peer_id=peer_id,
                        game_intro=game_intro_for_update,
                        game_history=game_history,
                        window_summaries=window_summaries_str,
                        current_ltm=current_ltm,
                        game_history_legend=game_history_legend
                    )
                for key, fut in future_opps.items():
                    opp_results[key] = fut.result()
            else:
                current_ltm = self.ltm_store.get(self.current_opponent_key) or "(No memory yet)"
                future_opp = ex.submit(
                    run_gradient_engine,
                    model=self.model,
                    game_intro=game_intro_for_update,
                    game_history=game_history,
                    window_summaries=window_summaries_str,
                    current_ltm=current_ltm,
                    game_history_legend=game_history_legend
                )
                opp_results[self.current_opponent_key] = future_opp.result()

            current_self_ltm = self.self_ltm_store.get("__self__") or "(No self-memory yet)"
            current_proactive_ltm = self.proactive_ltm_store.get("__overall__") or "(No proactive-memory yet)"
            player_index = getattr(self, 'current_player_index', None)
            agent_id = "You"
            
            future_self = ex.submit(
                run_self_gradient_engine,
                model=self.model,
                agent_id=agent_id,
                game_intro=game_intro_for_update,
                game_history=game_history,
                window_summaries=window_summaries_str,
                current_self_ltm=current_self_ltm,
                game_history_legend=game_history_legend
            )
            
            future_proactive = ex.submit(
                run_proactive_gradient_engine,
                model=self.model,
                agent_id=agent_id,
                game_intro=game_intro_for_update,
                game_history=game_history,
                window_summaries=window_summaries_str,
                current_proactive_ltm=current_proactive_ltm,
                game_history_legend=game_history_legend
            )
            
            self_structural_report, raw_self_grad_gen, self_grad_prompt = future_self.result()
            proactive_structural_report, raw_proactive_grad_gen, proactive_grad_prompt = future_proactive.result()

        for key, (rep, raw, prompt) in opp_results.items():
            self.logger.info(f'Gradient Report ({key}):\n{rep}')
        self.logger.info(f'Self Gradient Report:\n{self_structural_report}')
        self.logger.info(f'Proactive Gradient Report:\n{proactive_structural_report}')

        # Trace Logging
        log_file = getattr(self, '_parent_store_path', self.ltm_store_path).replace('.json', '_trace.log')
        with _trace_log_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                for key, (rep, raw, prompt) in opp_results.items():
                    f.write(f"=== GAME {getattr(self, 'game_count', 0)} POST-GAME GRADIENT REPORT ({key}) ===\n")
                    f.write(f"PROMPT:\n{prompt}\n")
                    f.write(f"RESPONSE (raw):\n{raw}\n")
                    f.write("=" * 50 + "\n\n")
    
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} SELF POST-GAME GRADIENT REPORT ===\n")
                f.write(f"PROMPT:\n{self_grad_prompt}\n")
                f.write(f"RESPONSE (raw):\n{raw_self_grad_gen}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} PROACTIVE POST-GAME GRADIENT REPORT ===\n")
                f.write(f"PROMPT:\n{proactive_grad_prompt}\n")
                f.write(f"RESPONSE (raw):\n{raw_proactive_grad_gen}\n")
                f.write("=" * 50 + "\n\n")

        if self.batch_mode:
            self._last_batch_result = {
                "opp": opp_results,
                "self": (self_structural_report, raw_self_grad_gen, self_grad_prompt),
                "proactive": (proactive_structural_report, raw_proactive_grad_gen, proactive_grad_prompt)
            }
            self.logger.info('Batch mode: opponent and self gradient data collected, synthesis deferred.')
            return

        # TGD Synthesis
        new_ltms = {}
        with ThreadPoolExecutor(max_workers=max(2, len(self.current_opponent_keys)+1)) as ex:
            if getattr(self, 'memory_mode', 'combined') == 'separate':
                from gamingbench.ltm.tgd_synthesizer import run_separate_tgd_synthesis
                future_synth = {}
                for key, (rep, raw, _) in opp_results.items():
                    peer_id = key.split(':')[0] if ':' in key else key
                    current_ltm = self.ltm_store.get(key) or "(No memory yet)"
                    future_synth[key] = ex.submit(
                        run_separate_tgd_synthesis,
                        model=self.model,
                        peer_id=peer_id,
                        game_intro=game_intro_for_update,
                        current_ltm=current_ltm,
                        gradient_reports=[rep]
                    )
                for key, fut in future_synth.items():
                    new_ltms[key] = fut.result()
            else:
                current_ltm = self.ltm_store.get(self.current_opponent_key) or "(No memory yet)"
                rep, raw, _ = opp_results[self.current_opponent_key]
                future_new_ltm = ex.submit(
                    run_tgd_synthesis,
                    model=self.model,
                    game_intro=game_intro_for_update,
                    current_ltm=current_ltm,
                    gradient_reports=[rep]
                )
                new_ltms[self.current_opponent_key] = future_new_ltm.result()

            future_new_self_ltm = ex.submit(
                run_self_tgd_synthesis,
                model=self.model,
                game_intro=game_intro_for_update,
                current_self_ltm=current_self_ltm,
                gradient_reports=[self_structural_report]
            )
            
            future_new_proactive_ltm = ex.submit(
                run_proactive_tgd_synthesis,
                model=self.model,
                game_intro=game_intro_for_update,
                current_proactive_ltm=current_proactive_ltm,
                gradient_reports=[proactive_structural_report]
            )
            
            new_self_ltm, raw_self_synth_gen, _ = future_new_self_ltm.result()
            new_proactive_ltm, raw_proactive_synth_gen, _ = future_new_proactive_ltm.result()

        for key, (new_ltm, raw_synth_gen, _) in new_ltms.items():
            self.logger.info(f'New LTM ({key}):\n{new_ltm}')
            self.ltm_store.update(key, new_ltm)
        self.ltm_store.save(self.ltm_store_path)

        self.logger.info(f'New Self LTM:\n{new_self_ltm}')
        self.self_ltm_store.update("__self__", new_self_ltm)
        self.self_ltm_store.save(self.self_ltm_store_path)
        
        self.logger.info(f'New Proactive LTM:\n{new_proactive_ltm}')
        self.proactive_ltm_store.update("__overall__", new_proactive_ltm)
        self.proactive_ltm_store.save(self.proactive_ltm_store_path)

    def flush_batch_updates(self, gradient_data: list) -> None:
        if not gradient_data:
            return

        n = len(gradient_data)
        opp_data_by_key = {}
        self_reports = []
        proactive_reports = []
        for d in gradient_data:
            if isinstance(d, dict):
                if "opp" in d:
                    for k, v in d["opp"].items():
                        if k not in opp_data_by_key:
                            opp_data_by_key[k] = []
                        opp_data_by_key[k].append(v)
                if "self" in d:
                    report = d["self"][0].strip()
                    if report:
                        self_reports.append(report)
                if "proactive" in d:
                    report = d["proactive"][0].strip()
                    if report:
                        proactive_reports.append(report)
            else:
                self.logger.warning("Old gradient format (non-dict) encountered in flush_batch_updates. Skipping.")

        if not opp_data_by_key and not self_reports and not proactive_reports:
            self.logger.warning("No valid gradient data extracted from batch payloads.")
            return

        self.logger.info(f'-' * 20 + f'Batch LTM Flush (N={n})' + '-' * 20)

        game_intro_for_update = self.current_game_intro or "Game rules unavailable."
        if getattr(self, 'hive_mode', False):
            from gamingbench.prompts.hive_prompts import HIVE_UPDATE_NOTICE
            game_intro_for_update = HIVE_UPDATE_NOTICE + "\n\n" + game_intro_for_update

        from concurrent.futures import ThreadPoolExecutor
        new_ltms = {}
        # Ensure max_workers is at least 2 and can handle all unique opponent keys
        with ThreadPoolExecutor(max_workers=max(2, len(opp_data_by_key) + 1)) as ex:
            if getattr(self, 'memory_mode', 'combined') == 'separate':
                from gamingbench.ltm.tgd_synthesizer import run_separate_tgd_synthesis
                future_synth = {}
                for key, reps_raws in opp_data_by_key.items():
                    structural_reports = [d[0].strip() for d in reps_raws if d[0].strip()]
                    if not structural_reports: continue
                    peer_id = key.split(':')[0] if ':' in key else key
                    current_ltm = self.ltm_store.get(key) or "(No memory yet)"
                    future_synth[key] = ex.submit(
                        run_separate_tgd_synthesis,
                        model=self.model,
                        peer_id=peer_id,
                        game_intro=game_intro_for_update,
                        current_ltm=current_ltm,
                        gradient_reports=structural_reports
                    )
                for key, fut in future_synth.items():
                    new_ltms[key] = fut.result()
            else:
                for key, reps_raws in opp_data_by_key.items():
                    structural_reports = [d[0].strip() for d in reps_raws if d[0].strip()]
                    if structural_reports:
                        current_ltm = self.ltm_store.get(key) or "(No memory yet)"
                        future_new_ltm = ex.submit(
                            run_tgd_synthesis,
                            model=self.model,
                            game_intro=game_intro_for_update,
                            current_ltm=current_ltm,
                            gradient_reports=structural_reports
                        )
                        new_ltms[key] = future_new_ltm.result()

            future_new_self_ltm = None
            if self_reports:
                current_self_ltm = self.self_ltm_store.get("__self__") or '(No self-memory yet)'
                future_new_self_ltm = ex.submit(
                    run_self_tgd_synthesis,
                    model=self.model,
                    game_intro=self.current_game_intro or 'Game rules unavailable.',
                    current_self_ltm=current_self_ltm,
                    gradient_reports=self_reports,
                )
                
            future_new_proactive_ltm = None
            if proactive_reports:
                current_proactive_ltm = self.proactive_ltm_store.get("__overall__") or '(No proactive-memory yet)'
                future_new_proactive_ltm = ex.submit(
                    run_proactive_tgd_synthesis,
                    model=self.model,
                    game_intro=self.current_game_intro or 'Game rules unavailable.',
                    current_proactive_ltm=current_proactive_ltm,
                    gradient_reports=proactive_reports,
                )
                
            new_self_ltm = None
            if future_new_self_ltm:
                new_self_ltm, raw_self_synth_gen, self_synth_prompt = future_new_self_ltm.result()
                
            new_proactive_ltm = None
            if future_new_proactive_ltm:
                new_proactive_ltm, raw_proactive_synth_gen, proactive_synth_prompt = future_new_proactive_ltm.result()

        log_file = getattr(self, '_parent_store_path', self.ltm_store_path).replace('.json', '_trace.log')
        with _trace_log_lock:
            with open(log_file, "a", encoding="utf-8") as f:
                for key, (new_ltm, raw_synth_gen, synth_prompt) in new_ltms.items():
                    self.logger.info(f'Batch New LTM ({key}):\n{new_ltm}')
                    self.ltm_store.update(key, new_ltm)
                    
                    f.write(f"=== BATCH LTM SYNTHESIS ({key}) ===\n")
                    f.write(f"PROMPT:\n{synth_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{raw_synth_gen}\n")
                    f.write("=" * 50 + "\n\n")

                self.ltm_store.save(self.ltm_store_path)
        
                if new_self_ltm:
                    self.logger.info(f'Batch New Self LTM:\n{new_self_ltm}')
                    self.self_ltm_store.update("__self__", new_self_ltm)
                    self.self_ltm_store.save(self.self_ltm_store_path)
                    
                    f.write(f"=== BATCH SELF LTM SYNTHESIS ===\n")
                    f.write(f"PROMPT:\n{self_synth_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{raw_self_synth_gen}\n")
                    f.write("=" * 50 + "\n\n")
                    
                if new_proactive_ltm:
                    self.logger.info(f'Batch New Proactive LTM:\n{new_proactive_ltm}')
                    self.proactive_ltm_store.update("__overall__", new_proactive_ltm)
                    self.proactive_ltm_store.save(self.proactive_ltm_store_path)
                    
                    f.write(f"=== BATCH PROACTIVE LTM SYNTHESIS ===\n")
                    f.write(f"PROMPT:\n{proactive_synth_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{raw_proactive_synth_gen}\n")
                    f.write("=" * 50 + "\n\n")
