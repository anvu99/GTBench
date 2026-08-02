import os
import copy
import threading
import concurrent.futures
from typing import List, Dict, Any, Tuple

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.ltm_rag_store import LTMRAGStore
from gamingbench.ltm.ltm_retriever import LTMRetriever
from gamingbench.ltm.rag_prompts import (
    LTM_INJECTION_PROMPT, SELF_LTM_INJECTION_PROMPT, PROACTIVE_LTM_INJECTION_PROMPT,
    WINDOW_SUMMARIZE_PROMPT, GRADIENT_ENGINE_PROMPT, TGD_SYNTHESIS_PROMPT,
    SELF_GRADIENT_ENGINE_PROMPT, SELF_TGD_SYNTHESIS_PROMPT,
    PROACTIVE_GRADIENT_ENGINE_PROMPT, PROACTIVE_TGD_SYNTHESIS_PROMPT
)
from gamingbench.ltm.gradient_engine import run_gradient_engine
from gamingbench.ltm.tgd_synthesizer import run_tgd_synthesis
from gamingbench.ltm.self_gradient_engine import run_self_gradient_engine
from gamingbench.ltm.self_tgd_synthesizer import run_self_tgd_synthesis
from gamingbench.ltm.proactive_gradient_engine import run_proactive_gradient_engine
from gamingbench.ltm.proactive_tgd_synthesizer import run_proactive_tgd_synthesis

_trace_log_lock = threading.Lock()

class LTMRAGAgent(PromptAgent):
    _embedder = None
    _embedder_lock = threading.Lock()

    @classmethod
    def _get_embedder(cls, model_name: str, gpu_id: int, max_length: int, instruction: str, use_flash_attn: bool):
        """
        Singleton pattern to initialize and fetch the PyTorch embedding model (e.g., Qwen).
        This ensures that across multiple agents in memory, we only load the heavy ML model onto the GPU once.
        """
        if cls._embedder is None:
            with cls._embedder_lock:
                if cls._embedder is None:
                    # Dynamically import to avoid loading heavy torch libraries if not using RAG
                    from gamingbench.ltm.qwen_embedder import QwenEmbedder
                    cls._embedder = QwenEmbedder(
                        model_name=model_name,
                        gpu_id=gpu_id,
                        max_length=max_length,
                        instruction=instruction,
                        use_flash_attn=use_flash_attn
                    )
        return cls._embedder

    def __init__(self, config, **kwargs):
        """
        Initializes the LTM RAG Agent.
        Sets up the embedder model (e.g., Qwen), initializes storage paths for the 
        three memory databases (Opponent, Self, Proactive), and loads existing data from disk.
        """
        super(LTMRAGAgent, self).__init__(config, **kwargs)
        
        # Frequency of how often the agent summarizes recent steps (e.g. every 5 moves)
        self.summarize_every = getattr(config, "summarize_every", 5)
        
        # Batch mode flag determines if we wait to flush memory updates all at once at the very end
        self.batch_mode = getattr(config, "batch_mode", False)
        
        # Load Qwen embedder configurations for retrieving text embeddings
        self.embed_model = getattr(config, "embed_model", "Qwen/Qwen3-Embedding-0.6B")
        self.embed_gpu_id = getattr(config, "embed_gpu_id", 0)
        self.embed_max_length = getattr(config, "embed_max_length", 8192)
        self.embed_instruction = getattr(config, "embed_instruction", "Given a game board state, retrieve the most relevant past game signals or strategies")
        self.embed_use_flash_attn = getattr(config, "embed_use_flash_attn", True)
        
        # Initialize the global, shared Qwen embedding model (using a singleton pattern to save GPU VRAM)
        self.embedder = self._get_embedder(
            self.embed_model, 
            self.embed_gpu_id, 
            self.embed_max_length, 
            self.embed_instruction, 
            self.embed_use_flash_attn
        )
        
        # Retrieval params controlling how strict/loose memory fetches are
        # Self-LTM uses a similarity threshold because it acts as a safety bounds-checker
        self.self_ltm_threshold = getattr(config, "self_ltm_threshold", 0.6)
        self.self_ltm_max = getattr(config, "self_ltm_max", 3)
        # Opponent and Proactive use top_k to grab the most definitively useful single strategy
        self.opp_top_k = getattr(config, "opp_top_k", 1)
        self.proac_top_k = getattr(config, "proac_top_k", 1)
        
        # Default storage paths for the three separate LTM JSON databases
        base_opp = getattr(config, "opp_store_path", "opp_ltm_rag.json")
        base_self = getattr(config, "self_store_path", "self_ltm_rag.json")
        base_proac = getattr(config, "proactive_store_path", "proactive_ltm_rag.json")
        
        self.opp_store_path = base_opp
        self.self_store_path = base_self
        self.proac_store_path = base_proac
            
        self.opp_store = LTMRAGStore()
        self.self_store = LTMRAGStore()
        self.proac_store = LTMRAGStore()
        
        if os.path.exists(self.opp_store_path): self.opp_store.load(self.opp_store_path)
        if os.path.exists(self.self_store_path): self.self_store.load(self.self_store_path)
        if os.path.exists(self.proac_store_path): self.proac_store.load(self.proac_store_path)
        
        self.current_opponent_key = None
        self.current_trajectory_observations = []
        self.current_trajectory_actions = []
        self.retrieval_log = {}
        self.window_summaries = []
        self.move_count = 0
        self.recent_internal_reasoning = []
        self._last_batch_result = None

    def set_storage_dir(self, storage_dir):
        """
        Updates the base storage directory for the LTM RAG databases.
        This is typically called when moving into a specific experiment folder, ensuring
        that the agent reads and writes its memories to the correct experiment-specific files.
        """
        # Extract the base file names to preserve across directory changes
        base_opp = os.path.basename(self.opp_store_path)
        base_self = os.path.basename(self.self_store_path)
        base_proac = os.path.basename(self.proac_store_path)
        
        # In separate memory mode (multi-agent environments), append the specific player ID so agents don't share memory files
        if getattr(self, 'memory_mode', 'combined') == 'separate' or getattr(self, 'hive_mode', False):
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base_opp:
                base_opp = base_opp.replace(".json", f"_{pid}.json")
                base_self = base_self.replace(".json", f"_{pid}.json")
                base_proac = base_proac.replace(".json", f"_{pid}.json")
                
        # Remap the storage paths to point inside the new experiment folder
        self.opp_store_path = os.path.join(storage_dir, base_opp)
        self.self_store_path = os.path.join(storage_dir, base_self)
        self.proac_store_path = os.path.join(storage_dir, base_proac)
        
        # Maintain backwards compatibility with older logging tools that expect `ltm_store_path`
        self.ltm_store_path = self.opp_store_path  
        
        # Load the JSON contents into the store objects from the new paths
        if os.path.exists(self.opp_store_path): self.opp_store.load(self.opp_store_path)
        if os.path.exists(self.self_store_path): self.self_store.load(self.self_store_path)
        if os.path.exists(self.proac_store_path): self.proac_store.load(self.proac_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        """
        Resets the internal tracking state at the start of a new game.
        Clears out the temporary trajectory logs (observations, actions), retrieval logs, 
        and reasoning traces from the previous game, and sets up the new opponent context.
        """
        # If we aren't in batch mode, we explicitly reload the store from disk to sync any concurrent updates made by other processes
        if not self.batch_mode:
            if os.path.exists(self.opp_store_path): self.opp_store.load(self.opp_store_path)
            if os.path.exists(self.self_store_path): self.self_store.load(self.self_store_path)
            if os.path.exists(self.proac_store_path): self.proac_store.load(self.proac_store_path)
            
        if not hasattr(self, 'game_count'):
            self.game_count = 0
        self.game_count += 1
        
        # Reset the volatile game-specific arrays
        self.current_trajectory_observations = []
        self.current_trajectory_actions = []
        self.round_chat_obs_idx = {}
        self.round_action_obs_idx = {}
        self.retrieval_log = {}
        self.window_summaries = []
        self.move_count = 0
        self._last_summary_idx = 0
        self.recent_internal_reasoning = []
        self.current_game_intro = game_intro
        
        if isinstance(opponent_key, list):
            self.current_opponent_keys = opponent_key
            self.memory_mode = 'separate'
            self.current_opponent_key = opponent_key[0] if len(opponent_key) == 1 else None
        else:
            self.current_opponent_keys = [opponent_key]
            self.memory_mode = 'combined'
            self.current_opponent_key = opponent_key

    def __deepcopy__(self, memo):
        """Prevent deepcopying the PyTorch embedder to avoid CUDA OOM."""
        import copy
        embedder = getattr(self, 'embedder', None)
        
        # Temporarily remove embedder to prevent it from being deepcopied
        # since PyTorch models and thread locks cannot be serialized or safely copied
        if hasattr(self, 'embedder'):
            delattr(self, 'embedder')
            
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
            
        # Restore the embedder to both the original object and the new copy
        if embedder is not None:
            self.embedder = embedder
            result.embedder = embedder
            
        return result

    def _build_prompts(self, observations):
        """
        Constructs the prompt for the current turn.
        In the LTM RAG architecture, this function acts as the In-Game Retrieval step.
        It evaluates the current board state against the three memory databases (Proactive, Opponent, Self)
        and injects any retrieved memory signals directly into the prompt so the LLM is aware of them.
        """
        # Get the default prompt strings from the base PromptAgent
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        # Construct a pure board state representation to use as the query for our vector database
        from gamingbench.prompts.observation_prompts import construct_observation_prompt
        board_state = construct_observation_prompt(observations, observations['env_name'])
        
        chat_context = observations.get('chat_context', '')
        state_str = board_state
        is_chat = getattr(self, '_in_chat_step', False)
        
        if getattr(self, 'enable_chat', False) and chat_context and chat_context != "No messages yet.":
            state_str = f"--- ONGOING CHAT ---\n{chat_context}\n\n" + board_state
            
        current_round = observations.get('game_round', len(self.current_trajectory_observations) + 1)
        phase_str = "Chat" if is_chat else "Action"
        
        # Track this board state in the agent's trajectory so we can anchor memories to it later
        self.current_trajectory_observations.append({
            "round": current_round,
            "phase": phase_str,
            "state": state_str
        })
        step_idx = len(self.current_trajectory_observations) - 1
        
        # Use semantic game round for logging to align with the game_history text
        current_round = observations.get('game_round', step_idx)
        
        # Determine if we should only retrieve 'Chat' signals or 'Action' signals
        is_chat = getattr(self, '_in_chat_step', False)
        type_filter = "Chat" if is_chat else "Action"

        # This list will hold all retrieved memory texts to be injected into the LLM context
        injections = []
        import re
        
        # 1. Proactive LTM (General Strategies)
        proac_signals = self.proac_store.get_signals("__overall__")
        if proac_signals:
            # We use top_k retrieval (usually k=1) to fetch the single most relevant strategy
            retriever = LTMRetriever(proac_signals, mode="top_k", top_k=self.proac_top_k, 
                                     embedder=self.embedder, type_filter=type_filter)
            retrieved = retriever.retrieve(board_state)
            if retrieved:
                for sig in retrieved:
                    # Log what we retrieved so we can include it in the window summaries later
                    entry = self.retrieval_log.setdefault(sig["name"], {"text": sig["text"], "steps": [], "source": "Proactive Strategy"})
                    entry["steps"].append((current_round, type_filter))
                
                # Format the retrieved texts into a single block, stripping the Type field since the agent doesn't need to see it
                texts = [re.sub(r'^\s*-\s*Type:\s*(?:Action|Chat)\s*\n', '', s["text"], flags=re.MULTILINE) for s in retrieved]
                text_blob = "\n\n---\n\n".join(texts)
                from gamingbench.ltm.rag_prompts import PROACTIVE_LTM_INJECTION_PROMPT
                injections.append(PROACTIVE_LTM_INJECTION_PROMPT.format(proactive_ltm_text=text_blob))

        # 2. Opponent LTM (Specific Opponent Exploits/Patterns)
        if self.current_opponent_key:
            opp_signals = self.opp_store.get_signals(self.current_opponent_key)
            if opp_signals:
                # Also uses top_k retrieval to find the most relevant opponent-specific pattern
                retriever = LTMRetriever(opp_signals, mode="top_k", top_k=self.opp_top_k, 
                                         embedder=self.embedder, type_filter=type_filter)
                retrieved = retriever.retrieve(board_state)
                if retrieved:
                    for sig in retrieved:
                        entry = self.retrieval_log.setdefault(sig["name"], {"text": sig["text"], "steps": [], "source": "Opponent Reputation"})
                        entry["steps"].append((current_round, type_filter))
                    texts = [re.sub(r'^\s*-\s*Type:\s*(?:Action|Chat)\s*\n', '', s["text"], flags=re.MULTILINE) for s in retrieved]
                    text_blob = "\n\n---\n\n".join(texts)
                    from gamingbench.ltm.rag_prompts import LTM_INJECTION_PROMPT
                    injections.append(LTM_INJECTION_PROMPT.format(ltm_text=text_blob))
                    
        # 3. Self LTM (Personal Risks and Flaws)
        self_signals = self.self_store.get_signals("__self__")
        if self_signals:
            # Self LTM uses a threshold mode because it's a safety check. 
            # It will retrieve *all* signals above a strict similarity threshold to prevent the agent from repeating past mistakes.
            retriever = LTMRetriever(self_signals, mode="threshold", threshold=self.self_ltm_threshold, 
                                     max_results=self.self_ltm_max, embedder=self.embedder, type_filter=type_filter)
            retrieved = retriever.retrieve(board_state)
            if retrieved:
                for sig in retrieved:
                    entry = self.retrieval_log.setdefault(sig["name"], {"text": sig["text"], "steps": [], "source": "Self Reputation"})
                    entry["steps"].append((current_round, type_filter))
                texts = [re.sub(r'^\s*-\s*Type:\s*(?:Action|Chat)\s*\n', '', s["text"], flags=re.MULTILINE) for s in retrieved]
                text_blob = "\n\n---\n\n".join(texts)
                from gamingbench.ltm.rag_prompts import SELF_LTM_INJECTION_PROMPT
                injections.append(SELF_LTM_INJECTION_PROMPT.format(self_ltm_text=text_blob))
                
        if injections:
            from gamingbench.ltm.rag_prompts import UNIFIED_MEMORY_PREAMBLE
            # Build the memory overview list for the preamble
            memory_overview_lines = []
            if any("PROACTIVE MEMORY" in inj for inj in injections):
                memory_overview_lines.append("\n  - Proactive Memory: The most direct strategies and objectives to actively achieve your main game target.")
            if any("REACTIVE MEMORY" in inj for inj in injections):
                memory_overview_lines.append("\n  - Reactive Memory: Strategies to defend against or counter specific behavioral patterns observed in this opponent.")
            if any("VERIFICATION MEMORY" in inj for inj in injections):
                memory_overview_lines.append("\n  - Verification Memory: Essential safety checks to verify you are not falling into your own recurring mistakes.")
            memory_overview = "".join(memory_overview_lines)
            preamble = UNIFIED_MEMORY_PREAMBLE.format(memory_overview=memory_overview)
            
            injection_str = preamble + "\n" + "\n\n".join(injections)
            from gamingbench.prompts.observation_prompts import construct_game_intro
            env_name = observations['env_name']
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
            # Inject after game_intro
            observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + injection_str, 1)


        action_type = "chat message" if is_chat else "move"
        step_prompt = (
            f"As you reason through your {action_type}, please generate a summary of your internal thinking "
            f"regarding the game state. Explicitly detail how you used any provided proactive and reactive memory "
            f"to form your strategy, and how you used the verification memory to check your candidate {action_type}.\n\n"
            f"[Final Decision] Conclude your final {action_type} (You will output this in the required format later)."
        )
        
        observation_prompt += "\n\n" + step_prompt

        return system_prompt, observation_prompt


    def construct_init_messages(self, system_prompt, user_prompt):
        # 1. Strip the conflicting instruction globally
        user_prompt = user_prompt.replace("Do NOT output internal reasoning.", "")
        user_prompt = user_prompt.replace("do NOT output internal reasoning.", "")
        user_prompt = user_prompt.replace("and do NOT output internal reasoning.", "")
        user_prompt = user_prompt.replace("do not output internal reasoning.", "")
        user_prompt = user_prompt.replace("Please return your answer without explanation!", "")
        
        # 2. Extract the CoT block
        import re
        cot_pattern = r"(As you reason through your.*?(?:\[Final Decision\][^\n]*))"
        match_cot = re.search(cot_pattern, user_prompt, re.DOTALL)
        
        # 3. Extract the Final Instruction block
        if getattr(self, "_in_chat_step", False):
            final_inst_pattern = r"(Before making your next game move.*?)$"
        else:
            final_inst_pattern = r"(Your output must be in the following format:.*?)$"
        match_final = re.search(final_inst_pattern, user_prompt, re.DOTALL)
        
        if match_cot and match_final:
            cot_text = match_cot.group(1).strip()
            final_text = match_final.group(1).strip()
            
            # Clean up redundant phrasing from final_text to avoid confusing the LLM
            final_text = re.sub(r"You must choose an legal action to set up advantages\.?\s*", "", final_text, flags=re.IGNORECASE)
            final_text = re.sub(r"Your output must be in the following format:\s*", "", final_text, flags=re.IGNORECASE)
            final_text = final_text.strip()
            
            # Remove both from their original positions
            user_prompt = user_prompt.replace(match_cot.group(0), "")
            user_prompt = user_prompt.replace(match_final.group(0), "")
            
            # Clean up excess newlines
            user_prompt = re.sub(r'\n{3,}', '\n\n', user_prompt)
            
            output_type = "chat message" if getattr(self, "_in_chat_step", False) else "action"
            # Construct the unified block
            unified_block = (
                f"{cot_text}\n\n"
                "After your reasoning concludes, you MUST output EXACTLY two things, in this specific order:\n"
                "1. A concise summary (max 4 sentences) describing the opponent's recent behavior and which memories influenced your decision, wrapped in <summary>...</summary> tags.\n"
                f"2. The final {output_type}, formatted exactly as required below:\n\n"
                f"{final_text}\n"
            )
            
            user_prompt = user_prompt.strip() + "\n\n" + unified_block
            
        return super().construct_init_messages(system_prompt, user_prompt)

    def chat_step(self, observations, chat_history_str: str):
        """
        Executes a chat turn for the agent.
        Extracts the inline window summary from the reasoning process and logs it.
        """
        # Temporarily flag that we are in a chat context so _build_prompts retrieves chat-specific memories
        # Set context flag to trigger chat-specific memory retrieval
        self._in_chat_step = True
        message, query = super().chat_step(observations, chat_history_str)
        self._in_chat_step = False
        
        # Extract the inline summary from the agent's generation and save it
        if query:
            if hasattr(query, 'llm_output') and query.llm_output:
                raw_resp = query.llm_output[-1] if isinstance(query.llm_output, list) else str(query.llm_output)
                import re
                matches = re.findall(r"<summary>(.*?)</summary>", raw_resp, re.DOTALL)
                if matches:
                    summary_text = matches[-1].strip()
                    curr_round = observations.get('game_round', self.move_count + 1)
                    if not hasattr(self, 'window_summaries'):
                        self.window_summaries = []
                    self.window_summaries.append(f"Round {curr_round} (Chat):\n{summary_text}")
        
        # Store the exact index of this round's observation so the Gradient Engine can anchor to it later
        current_round = observations.get('game_round')
        if current_round is not None:
            self.round_chat_obs_idx[current_round] = len(self.current_trajectory_observations) - 1
            
        # Track action for current step
        # Since _build_prompts advanced the trajectory length, we just append to actions
        if message:
            self.current_trajectory_actions.append(f"[Chat] {message}")
        else:
            self.current_trajectory_actions.append("[Chat] (No message)")
        return message, query

    def step(self, observations):
        """
        Executes an action turn for the agent.
        Extracts the inline window summary from the reasoning process and logs it.
        """
        self._in_chat_step = False
        self.move_count += 1
        
        query_list = []
        move, sub_query_list = super().step(observations)
        if sub_query_list:
            for q in sub_query_list:
                if hasattr(q, 'llm_output') and q.llm_output:
                    raw_resp = q.llm_output[-1] if isinstance(q.llm_output, list) else str(q.llm_output)
                    import re
                    matches = re.findall(r"<summary>(.*?)</summary>", raw_resp, re.DOTALL)
                    if matches:
                        summary_text = matches[-1].strip()
                        curr_round = observations.get('game_round', self.move_count)
                        if not hasattr(self, 'window_summaries'):
                            self.window_summaries = []
                        self.window_summaries.append(f"Round {curr_round} (Action):\n{summary_text}")
            query_list.extend(sub_query_list)
            
        current_round = observations.get('game_round')
        if current_round is not None:
            self.round_action_obs_idx[current_round] = len(self.current_trajectory_observations) - 1
            
        if move:
            self.current_trajectory_actions.append(f"[Move] {move}")
        else:
            self.current_trajectory_actions.append("[Move] (None)")
            
        return move, query_list

    def post_game_update(self, game_history, final_board_state, env_name):
        """
        Orchestrates the post-game reflection (Gradient Engine) and memory consolidation (Synthesis Engine).
        1. Invokes the gradient engines to propose additions, modifications, or removals of signals.
        2. Parses the proposed gradient reports to map new/modified signals to explicit game states (anchors).
        3. Invokes the synthesis engines to merge the proposals permanently into the textual databases.
        4. Links the newly synthesized text back to the extracted board states and embeds them into the retrieval vector space.
        """
        # 1. Safety check: Abort if not playing a real recorded match against a tracked opponent
        if not self.current_opponent_key:
            return None, None

        # 2. Format the game history log so the LLM recognizes its own actions as "You"
        player_index = getattr(self, 'current_player_index', None)
        if player_index is not None:
            game_history = game_history.replace(f"Player {player_index}", "You")
        
        # 3. No final window summarization needed since summaries are extracted inline during actions.
        
        # Combine all the smaller window summaries into one master summary string for the Gradient Engine
        summaries_text = "\n\n".join(self.window_summaries) if self.window_summaries else "No window summaries generated."
        
        game_intro = getattr(self, "current_game_intro", "")
        
        from gamingbench.prompts.observation_prompts import construct_game_history_legend
        try:
            game_history_legend = construct_game_history_legend(env_name)
        except Exception:
            game_history_legend = "Game history legend unavailable."
        
        # 1. GENERATE GRADIENTS (Parallel)
        # We spawn 3 threads to query the LLM to reflect on the game from 3 different perspectives simultaneously:
        # - Opponent: "What did the opponent do that we can exploit?"
        # - Self: "What mistakes did we make that we should avoid?"
        # - Proactive: "What general strategies worked well?"
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_opp = executor.submit(
                run_gradient_engine, self.model,
                game_intro, game_history, summaries_text, 
                self.opp_store.get_text(self.current_opponent_key, retrieval_log=self.retrieval_log), 
                game_history_legend,
                GRADIENT_ENGINE_PROMPT
            )
            future_self = executor.submit(
                run_self_gradient_engine, self.model,
                "__self__", game_intro, game_history, summaries_text, 
                self.self_store.get_text("__self__", retrieval_log=self.retrieval_log), 
                game_history_legend,
                SELF_GRADIENT_ENGINE_PROMPT
            )
            future_proac = executor.submit(
                run_proactive_gradient_engine, self.model,
                "__overall__", game_intro, game_history, summaries_text, 
                self.proac_store.get_text("__overall__", retrieval_log=self.retrieval_log), 
                game_history_legend,
                PROACTIVE_GRADIENT_ENGINE_PROMPT
            )
            
            opp_res, opp_raw, opp_prompt = future_opp.result()
            self_res, self_raw, self_prompt = future_self.result()
            proac_res, proac_raw, proac_prompt = future_proac.result()

        # Helper Function: Parses the Gradient Engine's generated text report to extract the exact Step/Round 
        # numbers where new strategies were discovered. It then looks up those rounds in our trajectory list 
        # to fetch the exact raw board state string. This links the LLM's high-level strategy text to a 
        # concrete board state vector for the RAG database to embed.
        def build_anchor_map(report_text):
            anchor_map = {}
            if not isinstance(report_text, str):
                return anchor_map
                
            # Parse the string for: - [TAG] Signal: Name \n  - Type: Type \n  - Step: Step
            import re
            
            # Split by top-level bullets starting with optional "- " then "[" to find each entry
            entries = re.split(r'\n(?=(?:-\s*)?\[)', "\n" + report_text)
            
            for entry_text in entries:
                entry_text = entry_text.strip()
                if not entry_text:
                    continue
                    
                # Match the tag and signal name (make the leading "- " optional)
                tag_match = re.match(r'(?:-\s*)?\[([A-Z]+)\]\s*(?:Signal|Strategy|Signals|Strategies):\s*(.+)', entry_text, re.IGNORECASE)
                if not tag_match:
                    continue
                    
                tag_str = f"[{tag_match.group(1).upper()}]"
                name_str = tag_match.group(2).strip()
                
                # Match Round (robustly handling multiple comma-separated rounds)
                # Now we capture the entire string after 'Round:' to process multiple rounds
                round_line_match = re.search(r'[-]?\s*Round:(.*)', entry_text, re.IGNORECASE)
                # Match Type/Phase (the prompt instructs the LLM to output 'Type: Chat|Action')
                type_match = re.search(r'[-]?\s*(?:Phase|Type):\s*(\w+)', entry_text, re.IGNORECASE)
                
                if tag_str in ["[ADD]", "[MODIFY]", "[MERGE]", "[KEEP]"]:
                    if round_line_match:
                        # Find all numbers in the string, which gives us a list of round strings
                        rounds = re.findall(r'\d+', round_line_match.group(1))
                        strat_type = type_match.group(1) if type_match else "Action"
                        
                        boards = []
                        actions = []
                        # Iterate over each round the agent reported and grab the corresponding board state
                        for s in rounds:
                            try:
                                round_num = int(s)
                                idx = None
                                if strat_type.lower() == "chat":
                                    idx = self.round_chat_obs_idx.get(round_num)
                                else:
                                    idx = self.round_action_obs_idx.get(round_num)
                                    if idx is None:
                                        idx = self.round_action_obs_idx.get(round_num - 1)
                                        
                                if idx is not None and 0 <= idx < len(self.current_trajectory_observations):
                                    obs_item = self.current_trajectory_observations[idx]
                                    boards.append(obs_item["state"] if isinstance(obs_item, dict) else obs_item)
                                    actions.append(self.current_trajectory_actions[idx] if idx < len(self.current_trajectory_actions) else "")
                            except ValueError:
                                pass
                        
                        if boards:
                            anchor_map[name_str] = {"boards": boards, "actions": actions, "tag": tag_str}
                            
                    if name_str not in anchor_map:
                        self.logger.warning(f"Failed to extract or resolve explicit round for signal '{name_str}'. Excluding this signal from anchor creation.")
                            
            return anchor_map

        # Execute the helper function on all three generated gradient reports to extract the board state anchors
        opp_anchors = build_anchor_map(opp_res)
        self_anchors = build_anchor_map(self_res)
        proac_anchors = build_anchor_map(proac_res)

        # Print the raw text reports out to the main console/logs
        # Log to main log file
        self.logger.info(f'Opponent Gradient Report ({self.current_opponent_key}):\n{opp_res}')
        self.logger.info(f'Self Gradient Report:\n{self_res}')
        self.logger.info(f'Proactive Gradient Report:\n{proac_res}')

        # Trace Logging (Gradient Engine)
        opp_log_file = getattr(self, '_parent_store_path', getattr(self, 'opp_store_path', 'default_opp.json')).replace('.json', '_trace.log')
        self_log_file = getattr(self, '_parent_self_store_path', getattr(self, 'self_store_path', 'default_self.json')).replace('.json', '_trace.log')
        proac_log_file = getattr(self, '_parent_proac_store_path', getattr(self, 'proac_store_path', 'default_proac.json')).replace('.json', '_trace.log')
        
        with _trace_log_lock:
            with open(opp_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} OPPONENT GRADIENT REPORT ({self.current_opponent_key}) ===\n")
                f.write(f"PROMPT:\n{opp_prompt}\n")
                f.write(f"RESPONSE (raw):\n{opp_raw}\n")
                f.write("=" * 50 + "\n\n")

            with open(self_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} SELF GRADIENT REPORT ===\n")
                f.write(f"PROMPT:\n{self_prompt}\n")
                f.write(f"RESPONSE (raw):\n{self_raw}\n")
                f.write("=" * 50 + "\n\n")

            with open(proac_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} PROACTIVE GRADIENT REPORT ===\n")
                f.write(f"PROMPT:\n{proac_prompt}\n")
                f.write(f"RESPONSE (raw):\n{proac_raw}\n")
                f.write("=" * 50 + "\n\n")

        if self.batch_mode:
            self._last_batch_result = {
                "opp": (opp_res, opp_raw, opp_prompt, opp_anchors),
                "self": (self_res, self_raw, self_prompt, self_anchors),
                "proactive": (proac_res, proac_raw, proac_prompt, proac_anchors),
                "opponent_key": getattr(self, 'current_opponent_key', None)
            }
            return opp_raw, None
            
        # 3. SYNTHESIS ENGINE (Merge Gradients into DB)
        # Now we take the gradients (proposals) and pass them to the Synthesis LLM to permanently merge them into the textual database.
        opp_ltm_new, opp_syn_raw, opp_syn_prompt = run_tgd_synthesis(
            self.model, game_intro, self.opp_store.get_text(self.current_opponent_key), [opp_res],
            TGD_SYNTHESIS_PROMPT
        )
        self_ltm_new, self_syn_raw, self_syn_prompt = run_self_tgd_synthesis(
            self.model, game_intro, self.self_store.get_text("__self__"), [self_res],
            SELF_TGD_SYNTHESIS_PROMPT
        )
        proac_ltm_new, proac_syn_raw, proac_syn_prompt = run_proactive_tgd_synthesis(
            self.model, game_intro, self.proac_store.get_text("__overall__"), [proac_res],
            PROACTIVE_TGD_SYNTHESIS_PROMPT
        )
        
        self._apply_synthesis(opp_ltm_new, opp_anchors, self.opp_store, self.current_opponent_key)
        self._apply_synthesis(self_ltm_new, self_anchors, self.self_store, "__self__")
        self._apply_synthesis(proac_ltm_new, proac_anchors, self.proac_store, "__overall__")
        
        # Log to main log file
        self.logger.info(f'New Opponent LTM ({self.current_opponent_key}):\n{opp_ltm_new}')
        self.logger.info(f'New Self LTM:\n{self_ltm_new}')
        self.logger.info(f'New Proactive LTM:\n{proac_ltm_new}')

        # Trace Logging (Synthesis Engine)
        opp_log_file = getattr(self, '_parent_store_path', getattr(self, 'opp_store_path', 'default_opp.json')).replace('.json', '_trace.log')
        self_log_file = getattr(self, '_parent_self_store_path', getattr(self, 'self_store_path', 'default_self.json')).replace('.json', '_trace.log')
        proac_log_file = getattr(self, '_parent_proac_store_path', getattr(self, 'proac_store_path', 'default_proac.json')).replace('.json', '_trace.log')
        
        with _trace_log_lock:
            with open(opp_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} OPPONENT LTM SYNTHESIS ({self.current_opponent_key}) ===\n")
                f.write(f"PROMPT:\n{opp_syn_prompt}\n")
                f.write(f"RESPONSE (raw):\n{opp_syn_raw}\n")
                f.write("=" * 50 + "\n\n")

            with open(self_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} SELF LTM SYNTHESIS ===\n")
                f.write(f"PROMPT:\n{self_syn_prompt}\n")
                f.write(f"RESPONSE (raw):\n{self_syn_raw}\n")
                f.write("=" * 50 + "\n\n")

            with open(proac_log_file, "a", encoding="utf-8") as f:
                f.write(f"=== GAME {getattr(self, 'game_count', 0)} PROACTIVE LTM SYNTHESIS ===\n")
                f.write(f"PROMPT:\n{proac_syn_prompt}\n")
                f.write(f"RESPONSE (raw):\n{proac_syn_raw}\n")
                f.write("=" * 50 + "\n\n")
        
        if self.opp_store_path != '/dev/null': self.opp_store.save(self.opp_store_path)
        if self.self_store_path != '/dev/null': self.self_store.save(self.self_store_path)
        if self.proac_store_path != '/dev/null': self.proac_store.save(self.proac_store_path)

        return opp_raw, None

    def _apply_synthesis(self, tgd_output: str, anchor_map: dict, store: LTMRAGStore, key: str):
        """
        Takes the output of a Synthesis Engine (TGD) and permanently updates the JSON store.
        Parses the text blocks for individual signals, maps them to the board states extracted from the 
        Gradient Engine using the `anchor_map`, and calculates new centroid vectors for embedding matching.
        """
        if not tgd_output:
            self.logger.warning("Synthesis output is empty. Retaining previous LTM.")
            return
            
        # --- 1. SEPARATE THE SYNTHESIS OUTPUT ---
        # The LLM's synthesis report (tgd_output) typically contains the active strategies, 
        # a "Graveyard" of failed strategies, and a final "[ACCEPTED]" block that logs renames/merges.
        import re
        parts = re.split(r'(?i)(?:\n|^)[-*\[=\s]*ACCEPTED[^\n]*\n+', tgd_output)
        db_text = parts[0].strip()
        accepted_text = parts[1].strip() if len(parts) > 1 else ""
        if not accepted_text:
            self.logger.warning(f"No [ACCEPTED] block found in synthesis output for {key}. Strategies may fail to inherit centroids.")
        
        # --- 2. RESOLVE MERGES AND RENAMES ---
        # The [ACCEPTED] block maps newly synthesized strategies back to older ones they might have replaced.
        # Example line: - "New Flank Strategy" <- "[ADD] Signal: Old Flank A [Game 1]", "[ADD] Signal: Old Flank B [Game 2]"
        # 
        # Regex explanation for `r'(?:[-*+•]\s*)?"(.*?)"\s*<-\s*(.*)'`:
        # - `(?:[-*+•]\s*)?` : Matches an OPTIONAL bullet point of any type (-, *, +, or •) followed by optional whitespace. This prevents the parser from breaking if the LLM hallucinates different bullet styles.
        # - `"(.*?)"`        : Captures the new synthesized strategy name inside the first pair of quotes.
        # - `\s*<-\s*`       : Matches the literal arrow `<-` surrounded by optional whitespace.
        # - `(.*)`           : Captures the remaining string (the list of old source names) to be parsed next.
        import re
        accepted_mapping = {}
        for line in accepted_text.split('\n'):
            match = re.search(r'(?:[-*+•]\s*)?"(.*?)"\s*<-\s*(.*)', line)
            if match:
                new_name = match.group(1)
                old_names_str = match.group(2)
                old_names = re.findall(r'"(.*?)"', old_names_str)
                accepted_mapping[new_name] = old_names
            else:
                # Fallback for older LLM prompt formats that didn't specify the exact merge source
                match = re.search(r'(?:[-*+•]\s*)?"(.*?)"', line)
                if match:
                    name = match.group(1)
                    if name not in accepted_mapping:
                        accepted_mapping[name] = [name]
                
        # --- 3. PARSE THE GRAVEYARD ---
        # The "Graveyard" tracks strategies that failed or were counter-exploited. 
        # We save this so the agent doesn't try to reinvent bad ideas.
        # The Graveyard section is split using a flexible regex to handle formatting variations.
        # It matches lines containing "GRAVEYARD" with optional formatting characters (like ***, ---, ===).
        gyard_split = re.split(r'(?i)(?:\n|^)[-*=:#\s]*GRAVEYARD[^\n]*\n+', db_text)
        main_db = gyard_split[0]
        graveyard = gyard_split[1].strip() if len(gyard_split) > 1 else ""
        
        store.update_graveyard(key, graveyard)
        
        # Parse the JSON-like or text blocks of the synthesized textual database.
        # 
        # Regex explanation for the chunk delimiter `r'\n(?=(?:[-*+•]\s*)?(?:Signal|Strategy Name|Strategy|Name):)'`:
        # - `\n`             : We only split on newlines, keeping chunks properly line-separated.
        # - `(?=...)`        : Positive Lookahead assertion. It means "split here only IF what follows matches this pattern", but crucially it DOES NOT consume the matched text. This ensures the bullet point and "Signal:" prefix remain attached to the resulting string chunk, rather than being deleted by the split operation.
        # - `(?:[-*+•]\s*)?` : Matches an OPTIONAL bullet point character (-, *, +, or •) followed by whitespace.
        # - `(?:Signal|...)` : Matches the various label prefixes the LLM might hallucinate.
        blocks = re.split(r'\n(?=(?:[-*+•]\s*)?(?:Check|Signal|Strategy Name|Strategy|Name):)', "\n" + main_db, flags=re.IGNORECASE)
            
        if "no signals currently stored" not in main_db.lower() and "no strategies currently stored" not in main_db.lower() and len([b for b in blocks if b.strip()]) == 0:
            self.logger.warning(f"Synthesis returned 0 parseable signals for {key}, but did not explicitly declare the database empty. Potential formatting hallucination.")
            
        signals = []
        for b in blocks:
            b = b.strip()
            if not b:
                continue
                
            match = re.match(r'(?:[-*+•]\s*)?(?:Check|Signal|Strategy Name|Strategy|Name):\s*(.*)', b, flags=re.IGNORECASE)
            if not match:
                continue
                
            name = match.group(1).strip()
            sig = {"name": name, "text": b}
            found_anchor = False
            
            # --- 4. ANCHOR ATTACHMENT (THE CRUCIAL STEP) ---
            # A textual strategy is useless for RAG if it isn't linked to a concrete board state.
            # Here, we look at the old strategy names (from the [ACCEPTED] block) and find their corresponding 
            # raw board states from the `anchor_map` (which was generated by the Gradient Engine).
            # This links the new text database entries back to the exact numerical board states they were derived from.
            if name in accepted_mapping:
                old_names = accepted_mapping[name]
                sig["anchor_boards"] = []
                sig["anchor_actions"] = []
                sig["tags"] = []
                for old_name in old_names:
                    if old_name in anchor_map:
                        # Extend the signal's board/action anchor lists with all board/actions extracted for this signal
                        sig["anchor_boards"].extend(anchor_map[old_name]["boards"])
                        sig["anchor_actions"].extend(anchor_map[old_name]["actions"])
                        # Duplicate the tag so that every anchored board state has a matching tag in the list
                        sig["tags"].extend([anchor_map[old_name]["tag"]] * len(anchor_map[old_name]["boards"]))
                        found_anchor = True
                        continue
                        
                    # Fallback 1: Try new_name + Game tag
                    import re
                    game_match = re.search(r'(\[Game \d+\])', old_name)
                    if game_match:
                        game_tag = game_match.group(1)
                        fallback_name = f"{name} {game_tag}"
                        if fallback_name in anchor_map:
                            sig["anchor_boards"].extend(anchor_map[fallback_name]["boards"])
                            sig["anchor_actions"].extend(anchor_map[fallback_name]["actions"])
                            sig["tags"].extend([anchor_map[fallback_name]["tag"]] * len(anchor_map[fallback_name]["boards"]))
                            found_anchor = True
                            continue
                            
                    # Fallback 2: Substring matching over all anchors
                    for anchor_key in anchor_map:
                        if game_match and game_match.group(1) not in anchor_key:
                            continue # Must match the same game
                        
                        # Clean both names by stripping LLM formatting prefixes like "[ADD] Signal: " and tags like "[Game X]".
                        # 
                        # Regex `r'^(?:\[?[A-Z]+\]?\s*)?\[?(?:Strategy|Signal|Strategies|Signals):\s*'` explanation:
                        # - `^`                    : Anchors the match strictly to the start of the string.
                        # - `(?:\[?[A-Z]+\]?\s*)?` : Matches optional gradient tags like "[ADD]", "ADD", or "[MODIFY]" followed by whitespace.
                        # - `\[?(?:Signal...):\s*` : Matches the literal "Signal:" label along with optional stray brackets.
                        clean_old = re.sub(r'^(?:\[?[A-Z]+\]?\s*)?\[?(?:Check|Strategy|Signal|Strategies|Signals):\s*', '', old_name, flags=re.IGNORECASE)
                        
                        # Regex `r'\s*\[Game \d+\]\s*'` explanation:
                        # Matches and removes game identifiers like " [Game 1]" along with any surrounding whitespace.
                        clean_old = re.sub(r'\s*\[Game \d+\]\s*', '', clean_old, flags=re.IGNORECASE)
                        clean_old = clean_old.strip(' []').lower()
                        
                        clean_anchor = re.sub(r'\s*\[Game \d+\]\s*', '', anchor_key, flags=re.IGNORECASE)
                        clean_anchor = clean_anchor.strip(' []').lower()
                        
                        if clean_old and clean_anchor and clean_old == clean_anchor:
                            sig["anchor_boards"].extend(anchor_map[anchor_key]["boards"])
                            sig["anchor_actions"].extend(anchor_map[anchor_key]["actions"])
                            sig["tags"].extend([anchor_map[anchor_key]["tag"]] * len(anchor_map[anchor_key]["boards"]))
                            found_anchor = True
                            break
            
            if not found_anchor:
                # Check if it's an old signal being retained verbatim or merged. 
                # If so, it already has centroids and doesn't need a polluted fallback anchor.
                is_old_signal = any(s["name"].lower() == name.lower() for s in store.get_signals(key))
                is_merged = name in accepted_mapping
                
                if is_old_signal or is_merged:
                    pass # Don't pollute with fallback anchor, just inherit old centroids later
                else:
                    self.logger.warning(f"Failed to find anchor mapping for '{name}' in {key}. Falling back to final game state.")
                    if len(self.current_trajectory_observations) > 0:
                        obs_item = self.current_trajectory_observations[-1]
                        sig["anchor_boards"] = [obs_item["state"] if isinstance(obs_item, dict) else obs_item]
                        sig["anchor_actions"] = [""]
                        sig["tags"] = ["[ADD]"]
            
            # Inherit old centroids/examples if the signal existed previously or was merged
            old_signals = store.get_signals(key)
            sig["centroids"] = []
            sig["examples"] = []
            
            source_names = accepted_mapping.get(name, [name])
            inherited_names = set()
            for source_name in source_names:
                import re
                clean_source = re.sub(r'^(?:\[?[A-Z]+\]?\s*)?\[?(?:Check|Strategy|Signal|Strategies|Signals):\s*', '', source_name, flags=re.IGNORECASE)
                clean_source = re.sub(r'\s*\[Game \d+\]\s*', '', clean_source, flags=re.IGNORECASE)
                clean_source = clean_source.strip(' []')
                
                # Split by '+' to properly inherit from all parent signals during a [MERGE]
                split_sources = [s.strip() for s in clean_source.split('+')]
                
                for split_src in split_sources:
                    old_sig = next((s for s in old_signals if s["name"].lower() == split_src.lower()), None)
                    if old_sig and old_sig["name"] not in inherited_names:
                        inherited_names.add(old_sig["name"])
                        import copy
                        if "centroids" in old_sig:
                            sig["centroids"].extend(copy.deepcopy(old_sig["centroids"]))
                        # DISABLED FOR TESTING: Do not propagate examples
                        # if "examples" in old_sig:
                        #     sig["examples"].extend(copy.deepcopy(old_sig["examples"]))
                    elif not old_sig:
                        if "[ADD]" not in source_name.upper() and "[GAME " not in source_name.upper():
                            self.logger.warning(f"Source signal '{split_src}' (parsed from raw LLM output '{source_name}') not found in old database. No historical centroids inherited for '{name}'.")
                        
            if len(sig["centroids"]) > 5:
                self.logger.info(f"Compressing centroids for '{name}' in {key} from {len(sig['centroids'])} down to 5.")
            while len(sig["centroids"]) > 5:
                store._merge_closest_centroids(sig["centroids"])
                
            # DISABLED FOR TESTING
            # if "examples" in sig and len(sig["examples"]) > 5:
            #     # Bounding examples: if a merge caused it to exceed the limit, keep the 5 most recent
            #     sig["examples"] = sig["examples"][-5:]
                
            if not sig.get("centroids"):
                sig.pop("centroids", None)
            if not sig.get("examples"):
                sig.pop("examples", None)
            
            signals.append(sig)
            
        store.update_signals(key, signals)
        
        # --- 5. VECTOR EMBEDDING ---
        # Finally, we iterate over the attached board states for each signal and compute their mathematical embeddings.
        for sig in signals:
            name = sig["name"]
            if "anchor_boards" in sig:
                for i in range(len(sig["anchor_boards"])):
                    anchor_board = sig["anchor_boards"][i]
                    anchor_action = sig["anchor_actions"][i]
                    tag = sig["tags"][i]
                    
                    # If the signal was newly added, modified, or merged, we take the raw string of the board state
                    # and pass it through our embedding model (e.g., Qwen). We save the resulting vector (centroid) 
                    # into our database. In future games, the agent will convert its current board into a vector and 
                    # calculate the cosine similarity against this centroid to retrieve this strategy.
                    if tag in ["[ADD]", "[MODIFY]", "[MERGE]"]:
                        # Embed using shared embedder (document side, so is_query=False)
                        # We strictly embed the raw board state to maintain symmetric retrieval.
                        vec = self.embedder.encode(anchor_board, is_query=False)
                        store.add_centroid(key, name, vec)
                    
                    # If the strategy was just retained without changes, we store the new board state as an 
                    # experiential example to reinforce and expand the memory's matching surface area.
                    elif tag == "[KEEP]":
                        store.add_example(key, name, anchor_board, anchor_action, self.embedder)
                
                # Clean up transient lists so they aren't serialized into the JSON
                sig.pop("anchor_boards", None)
                sig.pop("anchor_actions", None)
                sig.pop("tags", None)

    def flush_batch_updates(self, gradient_data: list) -> None:
        """
        Used in batch/parallel mode where multiple games are played simultaneously.
        Aggregates the gradient reports across multiple games and runs a single massive 
        Synthesis Engine pass to consolidate all of them at once.
        """
        if not gradient_data:
            return

        n = len(gradient_data)
        opp_reports = []
        opp_anchors_list = []
        self_reports = []
        self_anchors_list = []
        proac_reports = []
        proac_anchors_list = []
        # --- 1. AGGREGATING GRADIENT DATA ---
        # gradient_data is a list of dictionaries, where each dict represents the Gradient Engine's
        # analysis from ONE game. We loop through all the games and group the reports by category
        # (Opponent, Self, Proactive) so we can send them all to the Synthesis Engine at once.
        for d in gradient_data:
            if isinstance(d, dict):
                if "opp" in d:
                    rep, raw, prompt, anchors = d["opp"]
                    if rep.strip():
                        opp_reports.append(rep)
                        opp_anchors_list.append(anchors)
                if "self" in d:
                    rep, raw, prompt, anchors = d["self"]
                    if rep.strip():
                        self_reports.append(rep)
                        self_anchors_list.append(anchors)
                if "proactive" in d:
                    rep, raw, prompt, anchors = d["proactive"]
                    if rep.strip():
                        proac_reports.append(rep)
                        proac_anchors_list.append(anchors)
                if d.get("opponent_key"):
                    self.current_opponent_key = d["opponent_key"]

        if not opp_reports and not self_reports and not proac_reports:
            self.logger.warning("No valid gradient data extracted from batch payloads.")
            return

        self.logger.info(f'-' * 20 + f'Batch LTM Flush (N={n})' + '-' * 20)

        game_intro = getattr(self, "current_game_intro", "")

        # --- 2. PARALLEL SYNTHESIS EXECUTION ---
        # We fire off three separate API calls to the LLM concurrently (one for each strategy category).
        # We feed the LLM *all* the gradient reports from *all* the batch games at once.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            future_opp = None
            if opp_reports:
                future_opp = ex.submit(
                    run_tgd_synthesis,
                    self.model, game_intro, self.opp_store.get_text(self.current_opponent_key), opp_reports,
                    TGD_SYNTHESIS_PROMPT
                )
                
            future_self = None
            if self_reports:
                future_self = ex.submit(
                    run_self_tgd_synthesis,
                    self.model, game_intro, self.self_store.get_text("__self__"), self_reports,
                    SELF_TGD_SYNTHESIS_PROMPT
                )
                
            future_proac = None
            if proac_reports:
                future_proac = ex.submit(
                    run_proactive_tgd_synthesis,
                    self.model, game_intro, self.proac_store.get_text("__overall__"), proac_reports,
                    PROACTIVE_TGD_SYNTHESIS_PROMPT
                )

            opp_res = future_opp.result() if future_opp else (None, None, None)
            self_res = future_self.result() if future_self else (None, None, None)
            proac_res = future_proac.result() if future_proac else (None, None, None)

        # --- 3. MERGING ANCHORS ACROSS GAMES ---
        # Because we played multiple games, "Round 5" happened multiple times.
        # To avoid anchor collisions, we tag every anchor key with its specific Game ID.
        # This matches the Fallback 1 logic inside `_apply_synthesis`.
        def merge_anchors(anchors_list):
            merged = {}
            for i, anchors in enumerate(anchors_list):
                game_tag = f" [Game {i + 1}]"
                for k, v in anchors.items():
                    merged[k + game_tag] = v
            return merged
            
        # --- 4. APPLY SYNTHESIS AND SAVE ---
        # For each category, if the LLM successfully generated a new database, 
        # we parse it, attach the merged anchors, embed the vectors, and save it.
        opp_ltm_new, opp_syn_raw, opp_syn_prompt = opp_res
        if opp_ltm_new:
            merged_opp_anchors = merge_anchors(opp_anchors_list)
            self._apply_synthesis(opp_ltm_new, merged_opp_anchors, self.opp_store, self.current_opponent_key)
            self.logger.info(f'Batch New Opponent LTM:\n{opp_ltm_new}')

        self_ltm_new, self_syn_raw, self_syn_prompt = self_res
        if self_ltm_new:
            merged_self_anchors = merge_anchors(self_anchors_list)
            self._apply_synthesis(self_ltm_new, merged_self_anchors, self.self_store, "__self__")
            self.logger.info(f'Batch New Self LTM:\n{self_ltm_new}')

        proac_ltm_new, proac_syn_raw, proac_syn_prompt = proac_res
        if proac_ltm_new:
            merged_proac_anchors = merge_anchors(proac_anchors_list)
            self._apply_synthesis(proac_ltm_new, merged_proac_anchors, self.proac_store, "__overall__")
            self.logger.info(f'Batch New Proactive LTM:\n{proac_ltm_new}')

        # Trace Logging
        opp_log_file = getattr(self, '_parent_store_path', getattr(self, 'opp_store_path', 'default_opp.json')).replace('.json', '_trace.log')
        self_log_file = getattr(self, '_parent_self_store_path', getattr(self, 'self_store_path', 'default_self.json')).replace('.json', '_trace.log')
        proac_log_file = getattr(self, '_parent_proac_store_path', getattr(self, 'proac_store_path', 'default_proac.json')).replace('.json', '_trace.log')
        
        with _trace_log_lock:
            if opp_ltm_new:
                with open(opp_log_file, "a", encoding="utf-8") as f:
                    f.write(f"=== BATCH OPPONENT LTM SYNTHESIS ({self.current_opponent_key}) ===\n")
                    f.write(f"PROMPT:\n{opp_syn_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{opp_syn_raw}\n")
                    f.write("=" * 50 + "\n\n")

            if self_ltm_new:
                with open(self_log_file, "a", encoding="utf-8") as f:
                    f.write(f"=== BATCH SELF LTM SYNTHESIS ===\n")
                    f.write(f"PROMPT:\n{self_syn_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{self_syn_raw}\n")
                    f.write("=" * 50 + "\n\n")

            if proac_ltm_new:
                with open(proac_log_file, "a", encoding="utf-8") as f:
                    f.write(f"=== BATCH PROACTIVE LTM SYNTHESIS ===\n")
                    f.write(f"PROMPT:\n{proac_syn_prompt}\n")
                    f.write(f"RESPONSE (raw):\n{proac_syn_raw}\n")
                    f.write("=" * 50 + "\n\n")

        if self.opp_store_path != '/dev/null': self.opp_store.save(self.opp_store_path)
        if self.self_store_path != '/dev/null': self.self_store.save(self.self_store_path)
        if self.proac_store_path != '/dev/null': self.proac_store.save(self.proac_store_path)
