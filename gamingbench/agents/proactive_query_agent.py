import os
import re
import json
import threading
import concurrent.futures
import copy
import time
from typing import List, Dict, Any

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.two_layer_store import TwoLayerStore
from gamingbench.ltm.stat_pool import StatPool
from gamingbench.ltm.strategy_store import StrategyStore
from gamingbench.ltm.two_layer_prompts import (
    PQA_QUESTION_GEN_PROMPT,
    STAT_PROPOSAL_PROMPT,
    STAT_UPDATE_PROMPT,
    MEMORY_CONTENT_UPDATE_PROMPT,
    STAT_DEFINITION_PROMPT,
    NEW_MEMORY_FINALIZATION_PROMPT,
        ROUTE_AND_MODIFY_PROMPT,
    PROACTIVE_INJECTION_BLOCK,
    IN_GAME_ASSESSMENT_SUFFIX,
    IN_GAME_ASSESSMENT_SUFFIX_NO_STRATEGY,
    STRATEGY_INJECTION_BLOCK,
    STRATEGY_REFLECTION_EXTRACTION_PROMPT,
    STRATEGY_TREND_SYNTHESIS_PROMPT,
    STRATEGY_MERGE_PROMPT
)
from gamingbench.prompts.observation_prompts import construct_observation_prompt, construct_game_intro
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags

def extract_json_block(text: str) -> dict:
    """Safely extracts and parses a JSON block from LLM output, handling markdown wrappers."""
    text = text.strip()
    match = re.search(r'```json\s*(.*?)\s*```', text, re.IGNORECASE | re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end+1]
        else:
            json_str = "{}"
    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}

class ProactiveQueryAgent(PromptAgent):
    """
    Agent that proactively asks strategic questions in-game to guide its memory retrieval.
    It operates independently of EvidenceMemoryAgent, generating questions every round
    and extracting memory *only* from its question review process.
    """
    
    # Singleton embedder to share VRAM across batch clones
    _embedder = None
    _embedder_lock = threading.Lock()

    @classmethod
    def _get_embedder(cls, model_name: str, gpu_id: int, max_length: int, instruction: str, use_flash_attn: bool):
        """Thread-safe instantiation of the QwenEmbedder singleton."""
        if cls._embedder is None:
            with cls._embedder_lock:
                if cls._embedder is None:
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
        super().__init__(config, **kwargs)
        
        self.batch_mode = getattr(config, "batch_mode", False)
        
        self.embed_model = getattr(config, "embed_model", "Qwen/Qwen3-Embedding-0.6B")
        self.embed_gpu_id = getattr(config, "embed_gpu_id", 0)
        self.embed_max_length = getattr(config, "embed_max_length", 8192)
        self.embed_instruction = getattr(config, "embed_instruction", "Given a summary of the opponent's behavior, find the most applicable strategic memory")
        self.embed_use_flash_attn = getattr(config, "embed_use_flash_attn", True)
        
        self.embedder = self._get_embedder(
            self.embed_model, 
            self.embed_gpu_id, 
            self.embed_max_length, 
            self.embed_instruction, 
            self.embed_use_flash_attn
        )
        
        self.in_game_top_k = getattr(config, "in_game_top_k", 3)
        self.post_game_top_k = getattr(config, "post_game_top_k", 6)
        self.max_evidence_per_memory = getattr(config, "max_evidence_per_memory", 6)
        self.max_questions_per_step = getattr(config, "max_questions_per_step", 3)
        self.max_stats_per_memory = getattr(config, "max_stats_per_memory", 10)
        
        self.store_path = getattr(config, "two_layer_store_path", "two_layer_store.json")
        self.store = TwoLayerStore()
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)
            
        self.stat_pool_path = getattr(config, "stat_pool_path", "stat_pool.json")
        self.stat_pool = StatPool()
        if os.path.exists(self.stat_pool_path):
            self.stat_pool.load(self.stat_pool_path)
            
        self.use_strategy_memory = getattr(config, "use_strategy_memory", True)
        self.strategy_store_path = getattr(config, "strategy_store_path", "strategy_store.json")
        self.strategy_store = StrategyStore()
        if self.use_strategy_memory and os.path.exists(self.strategy_store_path):
            self.strategy_store.load(self.strategy_store_path)
            
        self.current_opponent_key = None
        self.current_game_intro = None
        self.move_count = 0
        self.question_log = []
        self.strategy_log = []
        self._last_batch_result = None
        self.match_working_memory = "No working memory established yet."
        
        # Tracks the questions asked and whether they were answered during the game
        self.question_log = []

    # Class-level lock for thread-safe file logging
    _log_lock = threading.Lock()

    def _log_prompt(self, phase_title, prompt, raw_answer):
        """Helper to log the prompts for debugging to both standard logger and a dedicated file."""
        log_str = (
            f"=== {phase_title} ===\n"
            f"PROMPT:\n{prompt}\n"
            f"RAW ANSWER:\n{raw_answer}\n"
            f"{'=' * (8 + len(phase_title))}\n\n"
        )
        if hasattr(self, 'logger'):
            self.logger.info(f"=== {phase_title} ===")
            self.logger.info(f"PROMPT:\n{prompt}")
            self.logger.info(f"RAW ANSWER:\n{raw_answer}")
            self.logger.info("=" * (8 + len(phase_title)))
        
        with self._log_lock:
            try:
                log_dir = os.path.dirname(self.store_path) if hasattr(self, 'store_path') else ""
                log_file = "ProactiveQueryAgent_question_processing.log"
                if log_dir and log_dir != '/dev/null':
                    log_file = os.path.join(log_dir, f"{self.agent_name}_question_processing.log")
                    
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(log_str)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Failed to write to question processing log: {e}")

    def set_storage_dir(self, storage_dir):
        """Updates the store path when the framework overrides the storage directory."""
        base = os.path.basename(self.store_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate' or getattr(self, 'hive_mode', False):
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in base:
                base = base.replace(".json", f"_{pid}.json")
        
        self.store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)
            
        stat_base = os.path.basename(self.stat_pool_path)
        if getattr(self, 'memory_mode', 'combined') == 'separate' or getattr(self, 'hive_mode', False):
            pid = getattr(self, 'player_id', 'pX')
            if f"_{pid}.json" not in stat_base:
                stat_base = stat_base.replace(".json", f"_{pid}.json")
        self.stat_pool_path = os.path.join(storage_dir, stat_base)
        if os.path.exists(self.stat_pool_path):
            self.stat_pool.load(self.stat_pool_path)

        strat_base = os.path.basename(self.strategy_store_path)
        self.strategy_store_path = os.path.join(storage_dir, strat_base)
        if os.path.exists(self.strategy_store_path):
            self.strategy_store.load(self.strategy_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        if not self.batch_mode:
            if os.path.exists(self.store_path): 
                self.store.load(self.store_path)
            if self.use_strategy_memory and os.path.exists(self.strategy_store_path):
                self.strategy_store.load(self.strategy_store_path)
            
        self.game_count = getattr(self, 'game_count', 0) + 1
        
        self.current_trajectory = []
        self.move_count = 0
        self.current_game_intro = game_intro
        
        if isinstance(opponent_key, list):
            self.current_opponent_keys = opponent_key
            self.memory_mode = 'separate'
            self.current_opponent_key = opponent_key[0] if len(opponent_key) == 1 else None
        else:
            self.current_opponent_keys = [opponent_key]
            self.memory_mode = 'combined'
            self.current_opponent_key = opponent_key
            
        self.question_log = []
        self.strategy_log = []

    def __deepcopy__(self, memo):
        import copy
        embedder = getattr(self, 'embedder', None)
        if hasattr(self, 'embedder'):
            delattr(self, 'embedder')
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        if embedder is not None:
            self.embedder = embedder
            result.embedder = embedder
        return result

    def _get_strategy_injection_block(self, env_name=None, top_score_k=7, top_recent_k=0, bottom_k=3):
        # 1. Fetch Best Strategies
        top_strats = self.strategy_store.get_top_k_by_score(top_k=top_score_k)
        top_ids = {s['id'] for s in top_strats}
        
        # 2. Fetch New Strategies (most recent that are not in best)
        all_strats = self.strategy_store.get_all()
        recent_strats = [s for s in all_strats if s['id'] not in top_ids]
        recent_strats.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        new_strats = recent_strats[:top_recent_k]
        new_ids = {s['id'] for s in new_strats}
        
        # 3. Fetch Worst Strategies (lowest utility, tested at least once, not in best or new)
        remaining_strats = [s for s in all_strats if s['id'] not in top_ids and s['id'] not in new_ids and s.get("uses_count", 0) > 0]
        remaining_strats.sort(key=lambda s: (s.get("average_utility", 0.0), -s.get("created_at", 0)))
        worst_strats = remaining_strats[:bottom_k]
        
        strat_lines = []
        
        def _format_strats(strats):
            if not strats: return
            for s in strats:
                avg_util = s.get('average_utility', 0.0)
                games = s.get('uses_count', 0)
                trend = s.get('recent_execution_log', 'No execution data yet. This strategy has not been tested.')
                if env_name == "liars_dice":
                    win_rate = (avg_util + 1) / 2
                    strat_lines.append(f"[{s['id']}] \"{s['title']}\" (Win Rate: {win_rate:.0%} | Games: {games})\nReasoning: {s['strategic_reasoning']}\nAction: {s['tactical_guidance']}\nRecent Execution Log: {trend}\n")
                else:
                    strat_lines.append(f"[{s['id']}] \"{s['title']}\" (Avg Utility: {avg_util:.2f} | Games: {games})\nReasoning: {s['strategic_reasoning']}\nAction: {s['tactical_guidance']}\nRecent Execution Log: {trend}\n")
                    
        _format_strats(top_strats)
        _format_strats(new_strats)
        _format_strats(worst_strats)
        
        if not strat_lines:
            strat_lines.append("No strategies available yet. You must create a new one.")
            
        return STRATEGY_INJECTION_BLOCK.format(strategy_list="\n".join(strat_lines))

    def _generate_summary_and_question(self, observations):
        """
        Runs the pre-action LLM call to summarize the state AND ask a strategic question.
        """
        sys_prompt, obs_prompt = PromptAgent._build_prompts(self, observations)
        
        # Fetch a mix of top-performing and recent memories for the prompt injection
        top_questions_text = "No prior top-performing questions available."
        if self.current_opponent_key:
            top_mems = self.store.get_mixed_top_k(self.current_opponent_key, top_score_k=6, top_recent_k=4)
            if top_mems:
                top_q_list = []
                for m in top_mems:
                    score = m.get("score", 0)
                    top_q_list.append(f"[{m['id']}] \"{m['question']}\"")
                top_questions_text = "\n".join(top_q_list)
                
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"--- CURRENT GAME STATE ---\n{obs_prompt}\n\n{PQA_QUESTION_GEN_PROMPT.format(top_questions=top_questions_text, working_memory=self.match_working_memory, max_questions=self.max_questions_per_step)}"}
        ]
        
        max_retries = 3
        summary = ""
        questions = []
        final_query = None
        
        for attempt in range(max_retries):
            responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            raw_resp = responses[0]
            stripped_resp = strip_thinking_block(raw_resp)
            
            if attempt == 0:
                self.logger.info("=== IN-GAME QUESTION GENERATION ===")
                self.logger.info(f"PROMPT:\n{messages[1]['content']}")
            self.logger.info(f"RAW ANSWER (Attempt {attempt+1}):\n{raw_resp}")
            self.logger.info(f"STRIPPED ANSWER (Attempt {attempt+1}):\n{stripped_resp}")
            self.logger.info("===================================")
            
            parsed_json = extract_json_block(stripped_resp)
            error_parts = []
            
            if not parsed_json:
                error_parts.append("Failed to parse JSON. Please ensure your output is strictly valid JSON without unescaped LaTeX slashes (e.g., use \\\\ge instead of \\ge) and is not cut off.")
            else:
                summary = parsed_json.get("summary", "")
                questions_raw = parsed_json.get("questions", [])
                
                questions = []
                if isinstance(questions_raw, list):
                    for q in questions_raw:
                        if isinstance(q, dict):
                            q_text = q.get("question", "")
                            src_id = q.get("source_memory_id")
                            if isinstance(src_id, str) and (not src_id.strip() or src_id.strip().lower() == "null"):
                                src_id = None
                            if q_text:
                                if src_id:
                                    original_mem = self.store.get_memory(self.current_opponent_key, src_id)
                                    if original_mem and 'question' in original_mem:
                                        q_text = original_mem['question']
                                    else:
                                        continue
                                questions.append({"question": q_text, "source_memory_id": src_id})
                
                if not questions:
                    q_text = parsed_json.get("question", "")
                    src_id = parsed_json.get("source_memory_id")
                    if isinstance(src_id, str) and (not src_id.strip() or src_id.strip().lower() == "null"):
                        src_id = None
                    if q_text:
                        if src_id:
                            original_mem = self.store.get_memory(self.current_opponent_key, src_id)
                            if original_mem and 'question' in original_mem:
                                q_text = original_mem['question']
                            else:
                                error_parts.append(f"source_memory_id '{src_id}' not found.")
                        if not error_parts:
                            questions.append({"question": q_text, "source_memory_id": src_id})
                            
            if not error_parts:
                final_query = query
                break
                
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": raw_resp})
                messages.append({"role": "user", "content": " ".join(error_parts) + " Please fix and try again."})
        
        return summary, questions, final_query
        
    def _run_in_game_memory_retrieval(self, summary_text, question_text, source_memory_id=None):
        """Helper to process the retrieval using both summary and question."""
        retrieved_mems = []
        injection = ""
        
        if self.current_opponent_key and question_text:
            if source_memory_id:
                # Retrieve directly if a source memory ID was provided
                direct_mem = self.store.get_memory(self.current_opponent_key, source_memory_id)
                if direct_mem:
                    retrieved_mems = [direct_mem]
                    
            if not retrieved_mems:
                # Fall back to semantic search if no source ID or direct lookup failed
                query_vec = self.embedder.encode(question_text, is_query=True)
                retrieved_mems = self.store.find_relevant_memories(self.current_opponent_key, query_vec, top_k=self.in_game_top_k)
            
            if retrieved_mems:
                mem_texts = [m['content'] for m in retrieved_mems]
                text_blob = "\n\n---\n\n".join(mem_texts)
            else:
                text_blob = "None."
                
        return retrieved_mems, None
        
    def _build_injection_block(self, all_retrieved_mems_list):
        """Helper to build a consolidated, labeled strategic memory injection block."""
        if not all_retrieved_mems_list:
            return ""
            
        questions_and_memories_blocks = []
        for idx, (q_dict, retrieved_mems) in enumerate(all_retrieved_mems_list):
            q_text = q_dict["question"]
            src_id = q_dict.get("source_memory_id")
            if src_id:
                mem_text = "\n".join([f"- [{m['id']}] {m['content']}" for m in retrieved_mems if m['id'] == src_id])
                if not mem_text:
                    mem_text = "\n".join([f"- [{m['id']}] {m['content']}" for m in retrieved_mems]) if retrieved_mems else "No relevant memories found."
                questions_and_memories_blocks.append(f"[Q{idx+1} - DIRECT RETRIEVAL] \"{q_text}\"\n{mem_text}\n")
            else:
                mem_text = "\n".join([f"- [{m['id']}] {m['content']}" for m in retrieved_mems]) if retrieved_mems else "No relevant memories found."
                questions_and_memories_blocks.append(f"[Q{idx+1} - NEW QUESTION] \"{q_text}\"\n{mem_text}\n")
            
        question_blocks_str = "\n".join(questions_and_memories_blocks)
        
        return PROACTIVE_INJECTION_BLOCK.format(question_blocks=question_blocks_str)

    def step(self, observations):
        """
        Main action loop for the proactive agent on its turn.
        
        The execution flow is:
        1. Proactive Query Generation: The agent looks at the current game state and generates
           a list of strategic questions it wants to ask its memory database (e.g., "Has the opponent bluffed in this situation before?").
        2. Memory Retrieval: For each question, it performs a semantic search against the TwoLayerStore
           to retrieve relevant long-term memories.
        3. Prompt Injection: It concatenates the retrieved memories and any overarching strategies
           into the observation prompt.
        4. Action Generation: It queries the LLM with this enriched prompt to choose the best legal move,
           and additionally assesses whether the retrieved memories were actually helpful.
        5. Trajectory Logging: It logs the state, questions, and memory assessments into `self.current_trajectory`
           which is later used by the batched post-game pipeline (`flush_batch_updates`).
        """
        self.move_count += 1
        query_list = []
        
        # 1. Proactive Query Generation
        summary_text, questions, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        # Log the current step for the post-game trajectory
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        
        # 2. Build prompt injections for all questions by iteratively retrieving memories
        all_retrieved_mems_list = []
        if not questions:
            self.current_trajectory.append({
                "round": observations.get('game_round', self.move_count),
                "phase": "Action",
                "state": board_state,
                "summary": summary_text,
                "question": ""
            })
            
        for q_dict in questions:
            q_text = q_dict["question"]
            src_id = q_dict["source_memory_id"]
            
            # Log each question individually into the trajectory for post-game tracking
            self.current_trajectory.append({
                "round": observations.get('game_round', self.move_count),
                "phase": "Action",
                "state": board_state,
                "summary": summary_text,
                "question": q_text
            })
            
            # Retrieve memories for this specific question (or direct fetch if src_id is provided)
            retrieved_mems, _ = self._run_in_game_memory_retrieval(
                summary_text, q_text, source_memory_id=src_id
            )
            # Store the retrieved memories to inject into the LLM prompt and to assess them later
            all_retrieved_mems_list.append((q_dict, retrieved_mems))
            
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        
        # 3. Prompt Injection: Insert strategies and retrieved memories directly above the board state
        if not self.match_working_memory or self.match_working_memory.strip() == "No working memory established yet.":
            wm_injection = ""
        else:
            wm_injection = f"=== WORKING MEMORY ===\n{self.match_working_memory}\n=====================\n"
            
        if self.use_strategy_memory:
            strat_injection = self._get_strategy_injection_block(env_name)
            full_injection = (wm_injection + "\n" + strat_injection) if wm_injection else strat_injection
        else:
            full_injection = wm_injection
        if all_retrieved_mems_list:
            combined_injection = self._build_injection_block(all_retrieved_mems_list)
            observation_prompt = observation_prompt.replace(board_state, full_injection + "\n\n" + combined_injection + "\n\n" + board_state, 1)
        else:
            observation_prompt = observation_prompt.replace(board_state, full_injection + "\n\n" + board_state, 1)

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead."
            
        step_prompt += IN_GAME_ASSESSMENT_SUFFIX if self.use_strategy_memory else IN_GAME_ASSESSMENT_SUFFIX_NO_STRATEGY
            
        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""
        batch_assessments = None
        
        for attempt in range(max_retries):
            responses, query = self.llm_query(msgs, n=self.num_generations, stop=None, prompt_type='move')
            query_list.append(query)
            
            if attempt == 0:
                self.logger.info(f'Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')
            
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Stripped Response (Attempt {attempt+1}):\n{stripped_response}')
            error_parts = []
            
            # 1. Validate the move
            moves = self.parse_with_regex(responses, regex)
            if len(moves) != 0:
                move = self.post_processing(moves, majority_vote=getattr(self, "voting", False))
                def clean_action(act):
                    return act.replace('<', '').replace('>', '').replace('*', '').strip()
                cleaned_move = clean_action(move)
                matched_valid_move = next((m for m in valid_moves if clean_action(m) == cleaned_move), None)
                
                if not valid_moves or matched_valid_move is not None:
                    if matched_valid_move is not None:
                        move = matched_valid_move
                else:
                    error_parts.append(f"Invalid move '{move}'. Your move must be one of the legal actions: {valid_moves}.")
            else:
                move = ""
                error_parts.append(f"Failed to extract a valid move format. Legal actions: {valid_moves}.")
                
            # 2. Validate the JSON assessment and strategy
            parsed_strategy = None
            try:
                parsed_json = extract_json_block(stripped_response)
                
                # Parse strategy (only when strategy memory is enabled)
                if self.use_strategy_memory:
                    strategy_raw = parsed_json.get("strategy", {})
                    strat_type = strategy_raw.get("type")
                    if strat_type == "follow":
                        strat_id = strategy_raw.get("strategy_id", "")
                        if isinstance(strat_id, str):
                            strat_id = strat_id.strip()
                        if not strat_id or not self.strategy_store.get_strategy(strat_id):
                            error_parts.append(f"strategy_id '{strat_id}' not found in strategy store. Use an existing ID or set type to 'new'.")
                        else:
                            parsed_strategy = strategy_raw
                    elif strat_type == "new":
                        required = ["title", "strategic_reasoning", "tactical_guidance", "desired_post_game_reflection"]
                        missing = [f for f in required if not strategy_raw.get(f, "").strip()]
                        if missing:
                            error_parts.append(f"New strategy is missing required fields: {missing}.")
                        else:
                            parsed_strategy = strategy_raw
                    else:
                        error_parts.append("strategy.type must be 'follow' or 'new'.")
                    
                assessments = parsed_json.get("assessments", [])
                expected_len = len(all_retrieved_mems_list)
                if len(assessments) != expected_len:
                    error_parts.append(f"Failed to extract the correct number of assessments. Expected {expected_len}, got {len(assessments)}.")
                else:
                    assessments_output = [None] * expected_len
                    for idx, item in enumerate(assessments):
                        q_idx = item.get("question_index")
                        if isinstance(q_idx, int) and 1 <= q_idx <= expected_len:
                            target_idx = q_idx - 1
                        else:
                            target_idx = idx
                            
                        src_id = all_retrieved_mems_list[target_idx][0].get("source_memory_id")
                        q_type = item.get("question_type", "new")
                        
                        if q_type == "direct" and not src_id:
                            q_type = "new"
                            
                        if q_type == "direct" or bool(src_id):
                            ans = True
                            mem_concl = ""
                            d_id = src_id
                        else:
                            ans = item.get("answered", False)
                            if isinstance(ans, str):
                                ans = ans.lower() == 'true' or ans.lower() == 'yes'
                            mem_concl = item.get("memory_conclusion", "")
                            d_id = item.get("driving_memory_id")
                            if isinstance(d_id, str) and (not d_id.strip() or d_id.strip().lower() == "null"):
                                d_id = None
                            
                            if not ans:
                                d_id = None
                                
                        d_info = item.get("desired_additional_info", "")
                        assessments_output[target_idx] = (ans, mem_concl, d_id, d_info)
                        
                    if None in assessments_output:
                        error_parts.append(f"Duplicate question_index provided. Ensure each assessment has a unique index from 1 to {expected_len}.")
                    else:
                        batch_assessments = assessments_output
                        
                new_wm = parsed_json.get("working_memory", "")
                if new_wm:
                    self.match_working_memory = new_wm
            except Exception as e:
                error_parts.append(f"Failed to parse JSON assessment response: {str(e)}.")
                
            if not error_parts:
                if self.use_strategy_memory and parsed_strategy:
                    self.strategy_log.append({
                        "turn": self.move_count,
                        "type": parsed_strategy["type"],
                        "strategy_id": parsed_strategy.get("strategy_id"),
                        "strategy_dict": parsed_strategy if parsed_strategy["type"] == "new" else None
                    })
                break
                
            if attempt < max_retries - 1:
                msgs.append({"role": "assistant", "content": raw_response})
                msgs.append({"role": "user", "content": " ".join(error_parts) + " Please try again."})
                
        # Fallback if assessments never parsed
        if batch_assessments is None:
            batch_assessments = [(False, "", None, "")] * len(all_retrieved_mems_list)
            self.logger.warning(f"In-game answer assessment failed after {max_retries} attempts.")
            
        if self.use_strategy_memory and ('parsed_strategy' not in locals() or parsed_strategy is None):
            self.strategy_log.append({
                "turn": self.move_count,
                "type": "follow",
                "strategy_id": None,
                "strategy_dict": None
            })
            
        self.current_trajectory[-1]["action"] = f"[Move] {move}"
                
        for idx, (q_dict, retrieved_mems) in enumerate(all_retrieved_mems_list):
            q_text = q_dict["question"]
            src_id = q_dict["source_memory_id"]
            
            answered, memory_conclusion, driving_memory_id, desired_additional_info = batch_assessments[idx]
            
            # Fallback: if a specific source memory was requested and the LLM found it helpful but forgot the ID
            if src_id and answered and not driving_memory_id:
                driving_memory_id = src_id
                
            mem_text = "\n".join([f"- [{m['id']}] {m['content']}" for m in retrieved_mems]) if retrieved_mems else ""
            
            # Log the question and its assessment into the question_log for post-game processing
            self.question_log.append({
                "round": observations.get('game_round', self.move_count),
                "question": q_text,
                "is_direct_retrieval": bool(src_id),
                "source_memory_id": src_id,
                "retrieved_memory_ids": [m["id"] for m in retrieved_mems],
                "retrieved_memories_text": mem_text if retrieved_mems else "No relevant memories found.",
                "answered": answered,
                "memory_conclusion": memory_conclusion,
                "driving_memory_id": driving_memory_id,
                "desired_additional_info": desired_additional_info
            })
        
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        """
        Custom chat step loop using proactive question generation.
        
        This functions identically to `step()`, but instead of choosing a game move,
        it answers chat-based conversational prompts. It still:
        1. Evaluates the conversation and generates questions.
        2. Retrieves relevant long-term memories.
        3. Injects them into the chat prompt context.
        4. Queries the LLM to generate a chat response.
        """
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        self.move_count += 1
        query_list = []
        observations['chat_context'] = chat_history_str
        
        # 1. Proactive Query Generation for Chat
        summary_text, questions, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        
        # Build prompt injections for all questions by iteratively retrieving memories
        all_retrieved_mems_list = []
        if not questions:
            self.current_trajectory.append({
                "round": observations.get('game_round', self.move_count),
                "phase": "Chat",
                "state": board_state,
                "summary": summary_text,
                "question": ""
            })
            
        for q_dict in questions:
            q_text = q_dict["question"]
            src_id = q_dict["source_memory_id"]
            
            # Log each question individually into the trajectory for post-game tracking
            self.current_trajectory.append({
                "round": observations.get('game_round', self.move_count),
                "phase": "Chat",
                "state": board_state,
                "summary": summary_text,
                "question": q_text
            })
            
            # Retrieve memories for this specific question (or direct fetch if src_id is provided)
            retrieved_mems, _ = self._run_in_game_memory_retrieval(
                summary_text, q_text, source_memory_id=src_id
            )
            # Store the retrieved memories to assess them later
            all_retrieved_mems_list.append((q_dict, retrieved_mems))
            
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        
        if not self.match_working_memory or self.match_working_memory.strip() == "No working memory established yet.":
            wm_injection = ""
        else:
            wm_injection = f"=== WORKING MEMORY ===\n{self.match_working_memory}\n=====================\n"
            
        if self.use_strategy_memory:
            strat_injection = self._get_strategy_injection_block(env_name)
            full_injection = (wm_injection + "\n" + strat_injection) if wm_injection else strat_injection
        else:
            full_injection = wm_injection
        if all_retrieved_mems_list:
            combined_injection = self._build_injection_block(all_retrieved_mems_list)
            observation_prompt = observation_prompt.replace(board_state, full_injection + "\n\n" + combined_injection + "\n\n" + board_state, 1)
        else:
            observation_prompt = observation_prompt.replace(board_state, full_injection + "\n\n" + board_state, 1)

        if env_name == 'cooperative_negotiation':
            from gamingbench.prompts.chat_prompts import COOP_CHAT_INSTRUCTION as instruction
        else:
            from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION as instruction
            
        observation_prompt = observation_prompt + '\n\n' + instruction
        
        observation_prompt += IN_GAME_ASSESSMENT_SUFFIX if self.use_strategy_memory else IN_GAME_ASSESSMENT_SUFFIX_NO_STRATEGY
            
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        max_retries = 3
        message = ""
        batch_assessments = None
        
        for attempt in range(max_retries):
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            query_list.append(query)
            
            if attempt == 0:
                self.logger.info(f'Chat Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Chat Raw Response (Attempt {attempt+1}): {responses}')
            
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Chat Stripped Response (Attempt {attempt+1}):\n{stripped_response}')
            error_parts = []
            
            message = strip_chat_tags(stripped_response).strip()
            
            if not message:
                error_parts.append("Failed to extract a valid chat message. You must output a non-empty message wrapped by <chat>...</chat>.")
                
            # Validate JSON assessment and strategy
            parsed_strategy = None
            try:
                parsed_json = extract_json_block(stripped_response)
                
                # Parse strategy (only when strategy memory is enabled)
                if self.use_strategy_memory:
                    strategy_raw = parsed_json.get("strategy", {})
                    strat_type = strategy_raw.get("type")
                    if strat_type == "follow":
                        strat_id = strategy_raw.get("strategy_id", "")
                        if isinstance(strat_id, str):
                            strat_id = strat_id.strip()
                        if not strat_id or not self.strategy_store.get_strategy(strat_id):
                            error_parts.append(f"strategy_id '{strat_id}' not found in strategy store. Use an existing ID or set type to 'new'.")
                        else:
                            parsed_strategy = strategy_raw
                    elif strat_type == "new":
                        required = ["title", "strategic_reasoning", "tactical_guidance", "desired_post_game_reflection"]
                        missing = [f for f in required if not strategy_raw.get(f, "").strip()]
                        if missing:
                            error_parts.append(f"New strategy is missing required fields: {missing}.")
                        else:
                            parsed_strategy = strategy_raw
                    else:
                        error_parts.append("strategy.type must be 'follow' or 'new'.")
                    
                assessments = parsed_json.get("assessments", [])
                expected_len = len(all_retrieved_mems_list)
                if len(assessments) != expected_len:
                    error_parts.append(f"Failed to extract the correct number of assessments. Expected {expected_len}, got {len(assessments)}.")
                else:
                    assessments_output = [None] * expected_len
                    for idx, item in enumerate(assessments):
                        q_idx = item.get("question_index")
                        if isinstance(q_idx, int) and 1 <= q_idx <= expected_len:
                            target_idx = q_idx - 1
                        else:
                            target_idx = idx
                            
                        src_id = all_retrieved_mems_list[target_idx][0].get("source_memory_id")
                        q_type = item.get("question_type", "new")
                        
                        if q_type == "direct" and not src_id:
                            q_type = "new"
                            
                        if q_type == "direct" or bool(src_id):
                            ans = True
                            mem_concl = ""
                            d_id = src_id
                        else:
                            ans = item.get("answered", False)
                            if isinstance(ans, str):
                                ans = ans.lower() == 'true' or ans.lower() == 'yes'
                            mem_concl = item.get("memory_conclusion", "")
                            d_id = item.get("driving_memory_id")
                            if isinstance(d_id, str) and (not d_id.strip() or d_id.strip().lower() == "null"):
                                d_id = None
                            
                            if not ans:
                                d_id = None
                                
                        d_info = item.get("desired_additional_info", "")
                        assessments_output[target_idx] = (ans, mem_concl, d_id, d_info)
                        
                    if None in assessments_output:
                        error_parts.append(f"Duplicate question_index provided. Ensure each assessment has a unique index from 1 to {expected_len}.")
                    else:
                        batch_assessments = assessments_output
                        
                new_wm = parsed_json.get("working_memory", "")
                if new_wm:
                    self.match_working_memory = new_wm
            except Exception as e:
                error_parts.append(f"Failed to parse JSON assessment response: {str(e)}.")
                
            if not error_parts:
                self.logger.info(f"Chat Generated: {message}")
                if self.use_strategy_memory and parsed_strategy:
                    self.strategy_log.append({
                        "turn": self.move_count,
                        "type": parsed_strategy["type"],
                        "strategy_id": parsed_strategy.get("strategy_id"),
                        "strategy_dict": parsed_strategy if parsed_strategy["type"] == "new" else None
                    })
                break
                
            if attempt < max_retries - 1:
                msgs.append({"role": "assistant", "content": raw_response})
                msgs.append({"role": "user", "content": " ".join(error_parts) + " Please fix and try again."})
                
        # Fallback if assessments never parsed
        if batch_assessments is None:
            batch_assessments = [(False, "", None, "")] * len(all_retrieved_mems_list)
            self.logger.warning(f"In-game answer assessment failed after {max_retries} attempts.")
            
        if self.use_strategy_memory and ('parsed_strategy' not in locals() or parsed_strategy is None):
            self.strategy_log.append({
                "turn": self.move_count,
                "type": "follow",
                "strategy_id": None,
                "strategy_dict": None
            })
            
        self.current_trajectory[-1]["action"] = f"[Chat] {message}"
                
        for idx, (q_dict, retrieved_mems) in enumerate(all_retrieved_mems_list):
            q_text = q_dict["question"]
            src_id = q_dict["source_memory_id"]
            
            answered, memory_conclusion, driving_memory_id, desired_additional_info = batch_assessments[idx]
            
            # Fallback: if a specific source memory was requested and the LLM found it helpful but forgot the ID
            if src_id and answered and not driving_memory_id:
                driving_memory_id = src_id
                
            mem_text = "\n".join([f"- [{m['id']}] {m['content']}" for m in retrieved_mems]) if retrieved_mems else ""
            self.question_log.append({
                "round": observations.get('game_round', self.move_count),
                "question": q_text,
                "is_direct_retrieval": bool(src_id),
                "source_memory_id": src_id,
                "retrieved_memory_ids": [m["id"] for m in retrieved_mems],
                "retrieved_memories_text": mem_text if retrieved_mems else "No relevant memories found.",
                "answered": answered,
                "memory_conclusion": memory_conclusion,
                "driving_memory_id": driving_memory_id,
                "desired_additional_info": desired_additional_info
            })
        
        return message, query_list[-1] if query_list else None

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called centrally post-game. Exclusively processes the question log."""
        # Ensure we actually have an opponent key to map memories against; if not, we can't save anything.
        if not self.current_opponent_key:
            return

        # Log that we are beginning the post-game review process for this specific agent.
        self.logger.info('-' * 20 + f'{self.agent_name} Post-Game Question Log Processing' + '-' * 20)
        
        # Identify the agent in the game history so the LLM knows which perspective it is evaluating from.
        if self.agent_name:
            game_history = game_history.replace(self.agent_name, f"{self.agent_name} (You)")

        # Create a shallow copy of the question log so that if we clear it later, we still retain the data for processing.
        question_log_data = self.question_log.copy()
        strategy_log_data = self.strategy_log.copy()
        
        # If batch_mode is True, we do NOT process the memory or write to disk right now.
        if self.batch_mode:
            # Instead, we package the game's data and return it to the main runner script.
            # The main script will collect these from all parallel games and feed them to `flush_batch_updates()` later.
            self._last_batch_result = {
                "opponent_key": self.current_opponent_key,
                "question_log": question_log_data,
                "strategy_log": strategy_log_data,
                "game_history": game_history
            }
            # Exit early so we don't accidentally update the memory bank concurrently.
            return

        # If batch_mode is False, we process the game immediately (Sequential mode).
        # First, run the Question Review LLM prompt to see if we can extract any new factual evidence from the game log.
        correction_tasks, synthesis_tasks = self._process_question_log(self.current_opponent_key, question_log_data, game_history)
        
        if correction_tasks:
            self._execute_batched_corrections(self.current_opponent_key, correction_tasks)
        if synthesis_tasks:
            self._execute_batched_synthesis(self.current_opponent_key, synthesis_tasks)
        
        # Finally, if we have a valid path to save to (meaning we aren't a temporary clone with /dev/null)...
        if self.store_path != '/dev/null':
            self.store.save(self.store_path)
            
        if self.use_strategy_memory:
            new_s = self._process_strategy_log(strategy_log_data, game_history)
            if new_s:
                self._merge_new_strategies(new_s)
                
            if self.strategy_store_path != '/dev/null':
                self.strategy_store.save(self.strategy_store_path)

    def _process_strategy_log(self, strategy_log, game_history):
        """Extracts reflections and updates trends for strategies at the end of a game/batch."""
        if not strategy_log:
            return []
            
        # 1. Deduplicate 'follow' strategies by ID
        seen_ids = set()
        followed_strats = []
        new_strats = []
        
        for entry in strategy_log:
            if entry["type"] == "follow":
                sid = entry.get("strategy_id")
                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    strat_obj = self.strategy_store.get_strategy(sid)
                    if strat_obj:
                        followed_strats.append(strat_obj)
            elif entry["type"] == "new":
                if entry.get("strategy_dict"):
                    new_strats.append(entry["strategy_dict"])
                    
        if not followed_strats and not new_strats:
            return []
            
        # 2. Extract Reflections (followed and new)
        reflections = self._extract_strategy_reflections(followed_strats, new_strats, game_history)
        
        # Extract utility from game_history
        import re
        utility = 0.0
        match = re.search(r"Game Outcome: Your (?:score|net chips)=([-\d.]+)", game_history)
        if not match:
            match = re.search(r"Game Outcome: Cooperative final score = ([-\d.]+)", game_history)
        if match:
            utility = float(match.group(1))
        
        # 3. Apply reflections and utility to followed strategies
        for s_id, observation in reflections.items():
            if s_id.startswith("strat_"):
                self.strategy_store.add_reflection(s_id, observation)
                self.strategy_store.update_utility(s_id, utility)
                
        # 4. Synthesize trends for followed strategies
        if followed_strats:
            self._synthesize_strategy_trends(followed_strats)
                
        # 5. Return new strategies with their reflection and utility
        new_strats_with_utility = []
        if new_strats:
            for idx, s in enumerate(new_strats):
                temp_id = f"temp_new_{idx}"
                obs = reflections.get(temp_id, "No reflection available.")
                new_strats_with_utility.append((s, obs, utility))
                
        return new_strats_with_utility

    def _extract_strategy_reflections(self, followed_strats, new_strats, game_history):
        strategies_to_reflect_text = ""
        for s in followed_strats:
            strategies_to_reflect_text += f"ID: {s['id']}\nTitle: {s['title']}\nReasoning: {s['strategic_reasoning']}\nGuidance: {s['tactical_guidance']}\nDesired Reflection: {s['desired_post_game_reflection']}\n\n"
            
        for idx, s in enumerate(new_strats):
            temp_id = f"temp_new_{idx}"
            strategies_to_reflect_text += f"ID: {temp_id}\nTitle: {s['title']}\nReasoning: {s['strategic_reasoning']}\nGuidance: {s['tactical_guidance']}\nDesired Reflection: {s['desired_post_game_reflection']}\n\n"
            
        messages = [
            {'role': 'system', 'content': "You are an expert game strategist evaluating the performance of game strategies."},
            {'role': 'user', 'content': STRATEGY_REFLECTION_EXTRACTION_PROMPT.format(strategies_to_reflect=strategies_to_reflect_text.strip(), game_trajectory=game_history)}
        ]
        
        expected_ids = set([s['id'] for s in followed_strats] + [f"temp_new_{idx}" for idx in range(len(new_strats))])
        
        reflections_map = {}
        max_retries = 3
        for attempt in range(max_retries):
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            
            if attempt == 0:
                self.logger.info(f'Strategy Reflection Prompt: {messages[1]["content"]}')
            self.logger.info(f'Strategy Reflection Raw Response (Attempt {attempt+1}): {responses}')
            
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Strategy Reflection Stripped Response (Attempt {attempt+1}):\n{stripped_response}')
            try:
                parsed_json = extract_json_block(stripped_response)
                
                log_dir = os.path.dirname(self.strategy_store_path)
                if log_dir and log_dir != '/dev/null':
                    s_log_file = os.path.join(log_dir, f"{self.agent_name}_strategy_processing.log")
                    try:
                        with open(s_log_file, "a") as f:
                            f.write("=== STRATEGY REFLECTION ===\n")
                            f.write(f"PROMPT:\n{messages[1]['content']}\n")
                            f.write(f"RAW ANSWER:\n{raw_response}\n")
                            f.write(f"STRIPPED ANSWER:\n{stripped_response}\n")
                            f.write("========================\n\n")
                    except Exception as e:
                        self.logger.error(f"Failed to write to strategy processing log: {e}")
                        
                reflections = parsed_json.get("reflections", [])
                for r in reflections:
                    s_id = r.get("strategy_id")
                    obs = r.get("reflection_observation", "").strip()
                    if s_id and obs:
                        reflections_map[s_id] = obs
                
                missing_ids = expected_ids - set(reflections_map.keys())
                if missing_ids:
                    raise ValueError(f"Missing reflections for strategies: {missing_ids}")
                    
                break
            except Exception as e:
                self.logger.warning(f"Failed to parse strategy reflections on attempt {attempt+1}: {str(e)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": f"Failed: {str(e)}. Please try again and ensure ALL strategies have a reflection."})
                
        return reflections_map

    def _synthesize_strategy_trends(self, followed_strats):
        valid_strats = []
        strategies_data_text = ""
        
        for s in followed_strats:
            # We fetch it freshly from store to ensure we have the newly pushed reflection
            strat_obj = self.strategy_store.get_strategy(s['id'])
            if not strat_obj or not strat_obj.get("recent_reflections"):
                continue
                
            reflections_text = ""
            for idx, obs in enumerate(strat_obj["recent_reflections"]):
                reflections_text += f"{idx+1}. {obs}\n"
                
            strategies_data_text += (
                f"--- Strategy ID: {strat_obj['id']} ---\n"
                f"Title: {strat_obj['title']}\n"
                f"Strategic Reasoning: {strat_obj['strategic_reasoning']}\n"
                f"Tactical Guidance: {strat_obj['tactical_guidance']}\n"
                f"Recent Reflections:\n{reflections_text}\n\n"
            )
            valid_strats.append(strat_obj['id'])
            
        if not valid_strats:
            return
            
        messages = [
            {'role': 'system', 'content': "You are an expert game strategist evaluating the trend of multiple strategies."},
            {'role': 'user', 'content': STRATEGY_TREND_SYNTHESIS_PROMPT.format(
                strategies_data=strategies_data_text.strip()
            )}
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            
            try:
                parsed_json = extract_json_block(stripped_response)
                trends = parsed_json.get("trends", [])
                
                processed_ids = set()
                for t in trends:
                    s_id = t.get("strategy_id")
                    summary = t.get("recent_execution_log", "").strip()
                    if s_id and summary:
                        self.strategy_store.update_execution_log(s_id, summary)
                        processed_ids.add(s_id)
                        
                log_dir = os.path.dirname(self.strategy_store_path)
                if log_dir and log_dir != '/dev/null':
                    s_log_file = os.path.join(log_dir, f"{self.agent_name}_strategy_processing.log")
                    try:
                        with open(s_log_file, "a") as f:
                            f.write("=== STRATEGY TREND SYNTHESIS (BATCHED) ===\n")
                            f.write(f"PROMPT:\n{messages[1]['content']}\n")
                            f.write(f"RAW ANSWER:\n{raw_response}\n")
                            f.write(f"STRIPPED ANSWER:\n{stripped_response}\n")
                            f.write("========================\n\n")
                    except Exception as e:
                        self.logger.error(f"Failed to write to strategy processing log: {e}")
                        
                missing = set(valid_strats) - processed_ids
                if missing:
                    raise ValueError(f"Missing execution logs for strategy IDs: {missing}")
                    
                break
            except Exception as e:
                self.logger.warning(f"Failed to parse batched strategy execution logs on attempt {attempt+1}: {str(e)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": f"Failed: {str(e)}. Please try again and ensure ALL strategies have a recent_execution_log."})

    def _merge_new_strategies(self, new_strats_with_scores):
        if not new_strats_with_scores:
            return
            
        new_strats_text = ""
        for idx, (s, score, utility) in enumerate(new_strats_with_scores):
            temp_id = f"temp_new_{idx}"
            new_strats_text += f"ID: {temp_id}\nTitle: {s['title']}\nScore: {score}\nReasoning: {s['strategic_reasoning']}\nGuidance: {s['tactical_guidance']}\n\n"
            
        messages = [
            {'role': 'system', 'content': "You are an expert game strategist managing a database of strategies."},
            {'role': 'user', 'content': STRATEGY_MERGE_PROMPT.format(new_strategies=new_strats_text.strip())}
        ]
        
        max_retries = 3
        kept_ids = set()
        for attempt in range(max_retries):
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            
            if attempt == 0:
                self.logger.info(f'Strategy Merge Prompt: {messages[1]["content"]}')
            self.logger.info(f'Strategy Merge Raw Response (Attempt {attempt+1}): {responses}')
            
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Strategy Merge Stripped Response (Attempt {attempt+1}):\n{stripped_response}')
            try:
                parsed_json = extract_json_block(stripped_response)
                
                log_dir = os.path.dirname(self.strategy_store_path)
                if log_dir and log_dir != '/dev/null':
                    s_log_file = os.path.join(log_dir, f"{self.agent_name}_strategy_processing.log")
                    try:
                        with open(s_log_file, "a") as f:
                            f.write("=== STRATEGY MERGE ===\n")
                            f.write(f"PROMPT:\n{messages[1]['content']}\n")
                            f.write(f"RAW ANSWER:\n{raw_response}\n")
                            f.write(f"STRIPPED ANSWER:\n{stripped_response}\n")
                            f.write("======================\n\n")
                    except Exception as e:
                        self.logger.error(f"Failed to write to strategy processing log: {e}")
                
                raw_keep = parsed_json.get("keep", [])
                if not isinstance(raw_keep, list):
                    raw_keep = [raw_keep]
                for k in raw_keep:
                    kept_ids.add(k)
                    
                for g in parsed_json.get("merge_groups", []):
                    keep_val = g.get("keep")
                    if keep_val is not None:
                        if isinstance(keep_val, list):
                            for k in keep_val:
                                kept_ids.add(k)
                        else:
                            kept_ids.add(keep_val)
                break
            except Exception as e:
                self.logger.warning(f"Failed to parse strategy merge on attempt {attempt+1}: {str(e)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": "Failed to parse JSON. Please try again with valid JSON format."})
                
        new_merged_strats = []
        for idx, (s, obs, utility) in enumerate(new_strats_with_scores):
            temp_id = f"temp_new_{idx}"
            if temp_id in kept_ids or not kept_ids:
                new_id = self.strategy_store.add_strategy(
                    title=s["title"],
                    strategic_reasoning=s["strategic_reasoning"],
                    tactical_guidance=s["tactical_guidance"],
                    desired_post_game_reflection=s["desired_post_game_reflection"]
                )
                self.strategy_store.add_reflection(new_id, obs)
                self.strategy_store.update_utility(new_id, utility)
                new_merged_strats.append(self.strategy_store.get_strategy(new_id))
                
        # Generate the initial Recent Execution Log for the newly saved strategies
        if new_merged_strats:
            self._synthesize_strategy_trends(new_merged_strats)
    def _run_phase_a(self, opponent_key, q_log, gh, game_rules):
        """
        Phase A: Propose new stats per game.
        Iterates over the question log of a single game.
        - Unanswered questions trigger proposals for brand new memories.
        - Existing memories that requested 'desired_additional_info' trigger proposals for new stats for that existing memory.
        Returns a PhaseAResult containing 'new_memories' and 'desired_infos'.
        """
        from gamingbench.ltm.two_layer_prompts import STAT_PROPOSAL_PROMPT
        
        unanswered_map = {}
        desired_infos_map = {}
        
        n_idx = 0
        for q_entry in q_log:
            drv_id = q_entry.get("driving_memory_id")
            if isinstance(drv_id, list): drv_id = drv_id[0] if drv_id else None
            is_dr = q_entry.get("is_direct_retrieval")
            if isinstance(is_dr, list): is_dr = is_dr[0] if is_dr else False
            dai = q_entry.get("desired_additional_info", "")
            if isinstance(dai, list): dai = dai[0] if dai else ""

            if not is_dr and not drv_id:
                q_text = q_entry.get("question", "")
                if isinstance(q_text, list): q_text = q_text[0] if q_text else ""
                q_id = f"N_{n_idx}"
                n_idx += 1
                if dai:
                    unanswered_map[q_id] = f"[{q_id}] Question: {q_text}\n  Desired Info: {dai}"
                else:
                    unanswered_map[q_id] = f"[{q_id}] {q_text}"
            elif drv_id and dai:
                q_entry["driving_memory_id"] = drv_id  # sanitize entry in case it's a list
                desired_infos_map[drv_id] = q_entry
                
        if not unanswered_map and not desired_infos_map:
            class PhaseAResult:
                new_memories = []
                desired_infos = []
                stat_proposals = []
            return PhaseAResult()
            
        unanswered_str = "[New Unanswered Questions]\n"
        unanswered_str += "\n".join(unanswered_map.values()) if unanswered_map else "No new unanswered questions."
        
        unanswered_str += "\n\n[Existing Memories Requesting Additional Info]\n"
        if desired_infos_map:
            for m_id, d in desired_infos_map.items():
                mem = self.store.get_memory(opponent_key, m_id)
                if mem:
                    unanswered_str += f"[{m_id}]\n"
                    unanswered_str += f"  Memory Question: {mem.get('question', '')}\n"
                    unanswered_str += f"  Memory Content: {mem.get('content', '')}\n"
                    unanswered_str += f"  Desired Additional Info: {d['desired_additional_info']}\n"
                else:
                    unanswered_str += f"[{m_id}] desired info: {d['desired_additional_info']}\n"
        else:
            unanswered_str += "No existing memories requested additional info."
                
        prompt = STAT_PROPOSAL_PROMPT.format(
            game_rules=game_rules,
            unanswered_questions=unanswered_str,
            game_trajectory=gh,
            max_stats_per_memory=self.max_stats_per_memory
        )
        
        msg = [{"role": "user", "content": prompt}]
        resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
        
        self._log_prompt("PHASE A: STAT PROPOSAL", prompt, resp[0])
        
        parsed = extract_json_block(strip_thinking_block(resp[0]))
        
        class PhaseAResult:
            stat_proposals = parsed.get("stat_proposals", [])
            
        return PhaseAResult()
    def _run_phase_a_5_merge(self, all_new_proposals, game_rules):
        """
        Phase A.5: Batched Memory Consolidation
        Takes all stat_proposals generated across the batch from Phase A.
        Asks the LLM to merge conceptually identical proposals into comprehensive ones
        and combine their proposed_stats.
        Returns the deduplicated list of proposals.
        """
        if len(all_new_proposals) <= 1:
            return all_new_proposals
            
        from gamingbench.ltm.two_layer_prompts import STAT_MEMORY_MERGE_PROMPT
        import json
        
        # Serialize the proposals for the prompt
        props_str = ""
        for i, p in enumerate(all_new_proposals):
            props_str += f"[{i}]\n{json.dumps(p, indent=2)}\n\n"
            
        prompt = STAT_MEMORY_MERGE_PROMPT.format(
            game_rules=game_rules,
            batched_proposals=props_str.strip(),
            max_stats_per_memory=self.max_stats_per_memory
        )
        
        msg = [{"role": "user", "content": prompt}]
        resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
        
        self._log_prompt("PHASE A.5: BATCHED MEMORY CONSOLIDATION", prompt, resp[0])
        
        parsed = extract_json_block(strip_thinking_block(resp[0]))
        
        raw_keep = parsed.get("keep", [])
        if not isinstance(raw_keep, list):
            raw_keep = [raw_keep]
        # Sanitize to flatten any nested lists that the LLM mistakenly outputs
        sanitized_keep = []
        for item in raw_keep:
            if isinstance(item, list):
                sanitized_keep.extend(item)
            else:
                sanitized_keep.append(item)
        keep_indices = set(sanitized_keep)
        
        for group in parsed.get("merge_groups", []):
            if "keep" in group:
                val = group["keep"]
                if isinstance(val, list):
                    for v in val:
                        keep_indices.add(v)
                else:
                    keep_indices.add(val)
                
        merged_proposals = []
        for idx in keep_indices:
            if isinstance(idx, int) and 0 <= idx < len(all_new_proposals):
                merged_proposals.append(all_new_proposals[idx])
            elif isinstance(idx, str) and idx.isdigit() and 0 <= int(idx) < len(all_new_proposals):
                merged_proposals.append(all_new_proposals[int(idx)])
                
        return merged_proposals


    def _run_phase_b(self, opponent_key, all_proposed_stats, game_rules):
        """
        Phase B: Batched Stat Definition.
        Takes all proposed stats from all games in the current batch (from Phase A).
        For each proposed stat, performs a semantic search against the StatPool.
        Makes ONE LLM call to decide whether each proposed stat should 'inherit' an existing stat
        or 'define_new' to instantiate a completely new one.
        Returns a mapping from the global list index to the final resolved stat_id.
        """
        if not all_proposed_stats:
            return {}
            
        from gamingbench.ltm.two_layer_prompts import STAT_DEFINITION_PROMPT
        
        global_resolved_stats = {}
        chunk_size = 20
        chunks = [all_proposed_stats[i:i + chunk_size] for i in range(0, len(all_proposed_stats), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            global_offset = chunk_idx * chunk_size
            batched_proposals = ""
            
            for local_i, prop in enumerate(chunk):
                sdesc = prop.get("description", "")
                spseudo = prop.get("pseudocode", "")
                if not isinstance(spseudo, str): spseudo = __import__('json').dumps(spseudo) if spseudo else ""
                stype = prop.get("type", "")
                if not sdesc or not stype: continue
                
                query_vec = self.embedder.encode(sdesc, is_query=True)
                candidate_ids = self.stat_pool.find_relevant_stats(opponent_key, query_vec, top_k=3, stat_type=stype)
                pool_summary = self.stat_pool.format_pool_summary(opponent_key, candidate_ids)
                
                batched_proposals += f"[{local_i}] Proposed Type: {stype}\nProposed Desc: {sdesc}\nCandidates:\n{pool_summary}\n\n"
                
            if not batched_proposals:
                continue
                
            prompt = STAT_DEFINITION_PROMPT.format(
                game_rules=game_rules,
                batched_proposals=batched_proposals.strip()
            )
            
            msg = [{"role": "user", "content": prompt}]
            resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
            
            self._log_prompt(f"PHASE B: BATCHED STAT DEFINITION (CHUNK {chunk_idx+1}/{len(chunks)})", prompt, resp[0])
            
            parsed = extract_json_block(strip_thinking_block(resp[0]))
            decisions = parsed.get("decisions", [])
            
            # Pass 1: Resolve 'inherit' and 'define_new'
            for dec in decisions:
                local_i = dec.get("local_idx")
                if local_i is None: continue
                try: local_i = int(local_i)
                except ValueError: continue
                
                if local_i < 0 or local_i >= len(chunk):
                    continue
                
                global_i = global_offset + local_i
                
                action = dec.get("action")
                if action == "inherit":
                    global_resolved_stats[global_i] = dec.get("stat_id")
                elif action == "define_new":
                    prop = chunk[local_i]
                    stype = prop.get("type")
                    sdesc = prop.get("description", "")
                    spseudo = prop.get("pseudocode", "")
                    if not isinstance(spseudo, str): spseudo = __import__('json').dumps(spseudo) if spseudo else ""
                    if stype and sdesc:
                        vec = self.embedder.encode(sdesc, is_query=False).tolist()
                        try:
                            new_id = self.stat_pool.add_stat(opponent_key, stype, sdesc, spseudo, vec)
                            global_resolved_stats[global_i] = new_id
                        except ValueError as e:
                            self.logger.warning(f"Failed to add stat: {e}")
                            
            # Pass 2: Resolve 'inherit_from_local' cross-references
            for dec in decisions:
                local_i = dec.get("local_idx")
                if local_i is None: continue
                try: local_i = int(local_i)
                except ValueError: continue
                
                if local_i < 0 or local_i >= len(chunk):
                    continue
                
                global_i = global_offset + local_i
                
                action = dec.get("action")
                if action == "inherit_from_local":
                    target_local_idx = dec.get("target_local_idx")
                    if target_local_idx is None: continue
                    try: target_local_idx = int(target_local_idx)
                    except ValueError: continue
                    
                    curr_target_local = target_local_idx
                    depth = 0
                    resolved_id = None
                    
                    while depth < 5:
                        curr_target_global = global_offset + curr_target_local
                        if curr_target_global in global_resolved_stats:
                            resolved_id = global_resolved_stats[curr_target_global]
                            break
                        
                        found_next = False
                        for d in decisions:
                            try:
                                if int(d.get("local_idx", -1)) == curr_target_local and d.get("action") == "inherit_from_local":
                                    curr_target_local = int(d.get("target_local_idx", -1))
                                    found_next = True
                                    break
                            except (ValueError, TypeError):
                                pass
                                
                        if not found_next or curr_target_local == -1:
                            break
                        depth += 1
                        
                    if resolved_id:
                        global_resolved_stats[global_i] = resolved_id
                    else:
                        try:
                            original_proposal = chunk[local_i]
                            stype = original_proposal.get("type")
                            sdesc = original_proposal.get("description", "")
                            spseudo = original_proposal.get("pseudocode", "")
                            if not isinstance(spseudo, str): spseudo = __import__('json').dumps(spseudo) if spseudo else ""
                            if stype and sdesc:
                                vec = self.embedder.encode(sdesc, is_query=False).tolist()
                                try:
                                    new_id = self.stat_pool.add_stat(opponent_key, stype, sdesc, spseudo, vec)
                                    global_resolved_stats[global_i] = new_id
                                except ValueError as e:
                                    self.logger.warning(f"Failed to add fallback stat: {e}")
                        except IndexError:
                            pass
                            
            # Ensure every input stat IN THIS CHUNK has a resolved ID, fallback to defining new
            for local_i, prop in enumerate(chunk):
                global_i = global_offset + local_i
                if global_i not in global_resolved_stats:
                    stype = prop.get("type", "")
                    sdesc = prop.get("description", "")
                    spseudo = prop.get("pseudocode", "")
                    if not isinstance(spseudo, str): spseudo = __import__('json').dumps(spseudo) if spseudo else ""
                    if stype and sdesc:
                        vec = self.embedder.encode(sdesc, is_query=False).tolist()
                        try:
                            new_id = self.stat_pool.add_stat(opponent_key, stype, sdesc, spseudo, vec)
                            global_resolved_stats[global_i] = new_id
                        except ValueError:
                            pass
                            
        return global_resolved_stats

    def _run_stat_update(self, opponent_key, gh, game_rules, existing_stat_ids, newly_resolved_ids):
        """
        Phase 1: Numerical Stat Update.
        Runs per game. Takes the list of ALL stat IDs that need updates 
        (both previously existing stats and the newly resolved stats from Phase B).
        The LLM receives the game trajectory and returns delta updates (counts, distributions, etc.).
        Applies these deltas to the StatPool.
        Returns a mapping of memory_id -> list of changed stat_ids to inform Phase C.
        """
        from gamingbench.ltm.two_layer_prompts import STAT_UPDATE_PROMPT
        
        all_stat_ids = list(set(existing_stat_ids + newly_resolved_ids))
        if not all_stat_ids:
            return {}
            
        chunk_size = 20
        chunks = [all_stat_ids[i:i + chunk_size] for i in range(0, len(all_stat_ids), chunk_size)]
        all_updates = []
        
        def process_chunk(chunk_ids, chunk_idx):
            stat_list_str = ""
            for sid in chunk_ids:
                if opponent_key in self.stat_pool.stats and sid in self.stat_pool.stats[opponent_key]:
                    stat = self.stat_pool.stats[opponent_key][sid]
                    current_vals = ", ".join([f"{k}={v}" for k,v in stat["storage"].items()])
                    sdesc_val = stat.get('description', '')
                    spseudo = stat.get('pseudocode', '')
                    stat_list_str += f"[{sid}] {stat['type']}:\n  Desc: {sdesc_val}\n  Logic: {spseudo}\n  Current: {current_vals}\n\n"
                    
            if not stat_list_str:
                return []
                
            prompt = STAT_UPDATE_PROMPT.format(
                game_rules=game_rules,
                game_trajectory=gh,
                stat_list=stat_list_str.strip()
            )
            
            msg = [{"role": "user", "content": prompt}]
            resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
            
            self._log_prompt(f"PHASE 1: NUMERICAL STAT UPDATE (CHUNK {chunk_idx+1}/{len(chunks)})", prompt, resp[0])
            
            parsed = extract_json_block(strip_thinking_block(resp[0]))
            return parsed.get("stat_updates", [])
            
        if chunks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
                futures = [executor.submit(process_chunk, c, i) for i, c in enumerate(chunks)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_updates.extend(future.result())
                    except Exception as e:
                        self.logger.error(f"Error processing Phase 1 stat update chunk: {e}")
                        
        self.stat_pool.update_stats(opponent_key, "batch", all_updates)
        
        changed_stat_info = {}
        for upd in all_updates:
            sid = upd.get("stat_id")
            deltas = upd.get("deltas", {})
            is_non_zero = False
            for k, v in deltas.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if v2 not in (0, 0.0, None, "", []): is_non_zero = True
                else:
                    if v not in (0, 0.0, None, "", []): is_non_zero = True
                    
            if is_non_zero:
                if opponent_key in self.stat_pool.stats and sid in self.stat_pool.stats[opponent_key]:
                    refs = self.stat_pool.stats[opponent_key][sid].get("referenced_by", [])
                    for mem_id in refs:
                        if mem_id not in changed_stat_info:
                            changed_stat_info[mem_id] = []
                        changed_stat_info[mem_id].append(sid)
                        
        return changed_stat_info

    def _run_memory_content_update(self, opponent_key, game_rules, changed_stat_info, desired_infos, resolved_stats_map, all_proposed_stats):
        """
        Phase C: Existing Memory Content Update.
        Updates the textual content of existing memories.
        Triggered if:
        1. The memory has stats whose numerical values just changed in Phase 1 (across all games in the batch).
        2. The memory requested 'desired_additional_info' in Phase A.
        The LLM also manages stat eviction if the memory exceeds the 10-stat cap by
        returning an 'evict_stat_ids' list.
        """
        from gamingbench.ltm.two_layer_prompts import MEMORY_CONTENT_UPDATE_PROMPT
        
        desired_map = {}
        desired_stat_additions = {}
        
        for d in desired_infos:
            mid = d.get("memory_id")
            if not mid: continue
            
            info_req = d.get("desired_additional_info")
            if not info_req:
                info_req = "Detailed investigation requested."
                
            if mid not in desired_map:
                desired_map[mid] = []
            if info_req not in desired_map[mid]:
                desired_map[mid].append(info_req)
            
            added = []
            for p_stat in d.get("proposed_stats", []):
                for i, glob_p in enumerate(all_proposed_stats):
                    if glob_p is p_stat and i in resolved_stats_map:
                        added.append(resolved_stats_map[i])
            if added:
                if mid not in desired_stat_additions:
                    desired_stat_additions[mid] = []
                desired_stat_additions[mid].extend(added)
                
        all_mems = set(list(changed_stat_info.keys()) + list(desired_map.keys()))
        if not all_mems:
            return
            
        mem_blocks = []
        all_mems_list = list(all_mems)
        chunk_size = 10
        chunks = [all_mems_list[i:i + chunk_size] for i in range(0, len(all_mems_list), chunk_size)]
        
        all_content_updates = []
        
        def process_chunk(chunk_mems, chunk_idx):
            c_mem_blocks = []
            for m_id in chunk_mems:
                mem = self.store.get_memory(opponent_key, m_id)
                if not mem: continue
                
                q = mem.get("question", "Unknown")
                content = mem.get("content", "")
                stat_ids = mem.get("stat_ids", [])
                
                new_additions = list(dict.fromkeys(desired_stat_additions.get(m_id, [])))
                unique_total_stats = set(stat_ids + new_additions)
                total_after = len(unique_total_stats)
                
                changed_for_this = changed_stat_info.get(m_id, [])
                unchanged = [s for s in stat_ids if s not in changed_for_this]
                
                block = f"[{m_id}] Question: \"{q}\"\nCurrent content: \"{content}\"\n"
                if changed_for_this:
                    block += f"Updated stats: {self.stat_pool.format_for_injection(opponent_key, changed_for_this)}\n"
                if unchanged:
                    block += f"Unchanged stats: {self.stat_pool.format_for_injection(opponent_key, unchanged)}\n"
                if m_id in desired_map:
                    valid_infos = [str(x) for x in desired_map[m_id] if x]
                    if valid_infos:
                        joined_info = " | ".join(valid_infos)
                        block += f"Requested additional info: {joined_info}\n"
                if new_additions:
                    block += f"Newly added trackers for this info: {self.stat_pool.format_for_injection(opponent_key, new_additions)}\n"
                
                block += f"Total statistical trackers attached to this memory: {total_after}\n"
                if total_after > self.max_stats_per_memory:
                    block += f"WARNING: You are tracking {total_after} stats, exceeding the limit of {self.max_stats_per_memory}! You MUST evict at least {total_after - self.max_stats_per_memory} stats via `evict_stat_ids`.\n"
                c_mem_blocks.append(block)
                
            if not c_mem_blocks:
                return []
                
            prompt = MEMORY_CONTENT_UPDATE_PROMPT.format(
                game_rules=game_rules,
                changed_memories="\n\n".join(c_mem_blocks),
                max_stats_per_memory=self.max_stats_per_memory
            )
            
            msg = [{"role": "user", "content": prompt}]
            resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
            
            self._log_prompt(f"PHASE C: EXISTING MEMORY CONTENT UPDATE (CHUNK {chunk_idx+1}/{len(chunks)})", prompt, resp[0])
            
            parsed = extract_json_block(strip_thinking_block(resp[0]))
            return parsed.get("content_updates", [])
            
        if chunks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(chunks))) as executor:
                futures = [executor.submit(process_chunk, c, i) for i, c in enumerate(chunks)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_content_updates.extend(future.result())
                    except Exception as e:
                        self.logger.error(f"Error processing Phase C memory update chunk: {e}")
        
        for upd in all_content_updates:
            m_id = upd.get("memory_id")
            if not m_id: continue
            mem = self.store.get_memory(opponent_key, m_id)
            if not mem: continue
            
            if upd.get("update") and upd.get("new_content"):
                mem["content"] = upd["new_content"]
                vec = self.embedder.encode(upd["new_content"], is_query=False)
                mem["vec"] = vec.tolist()
                mem["generation"] = mem.get("generation", 0) + 1
                mem["updated_at"] = time.time()
                
            new_adds = list(dict.fromkeys(desired_stat_additions.get(m_id, [])))
            for sid in new_adds:
                if sid not in mem.get("stat_ids", []):
                    mem.setdefault("stat_ids", []).append(sid)
                    self.stat_pool.add_reference(opponent_key, sid, m_id)
                    
            evict_ids = upd.get("evict_stat_ids") or []
            for ev in evict_ids:
                if ev in mem["stat_ids"] and ev not in new_adds:
                    mem["stat_ids"].remove(ev)
                    self.stat_pool.remove_reference(opponent_key, ev, m_id)

    def _run_new_memory_finalization(self, opponent_key, new_questions, resolved_stats_map, all_proposed_stats, game_rules):
        """
        Phase D: New Memory Finalization.
        Finalizes brand new memories proposed in Phase A.
        Provides the LLM with the new questions and the fully resolved newly created stats (with initial values).
        The LLM outputs the finalized textual content, ensuring the text aligns with the assigned trackers.
        The finalized memory is then saved to the TwoLayerStore.
        """
        import uuid
        if not new_questions:
            return
            
        from gamingbench.ltm.two_layer_prompts import NEW_MEMORY_FINALIZATION_PROMPT
        
        mem_blocks = []
        mem_map = {}
        for i, draft in enumerate(new_questions):
            q = draft.get("question")
            dai = draft.get("desired_info")
            
            added = []
            for p_stat in draft.get("proposed_stats", []):
                for glob_idx, glob_p in enumerate(all_proposed_stats):
                    if glob_p is p_stat and glob_idx in resolved_stats_map:
                        added.append(resolved_stats_map[glob_idx])
            
            added = list(dict.fromkeys(added))
            mem_map[i] = {"draft": draft, "added": added}
            
            block = f"[Question {i}] \"{q}\"\n"
            if dai:
                block += f"Desired Info: \"{dai}\"\n"
            if added:
                block += f"Resolved trackers with initial values:\n{self.stat_pool.format_for_injection(opponent_key, added)}\n"
            block += f"Total statistical trackers attached to this memory: {len(added)}\n"
            if len(added) > self.max_stats_per_memory:
                block += f"WARNING: You are tracking {len(added)} stats, exceeding the limit of {self.max_stats_per_memory}! You MUST evict at least {len(added) - self.max_stats_per_memory} stats via `evict_stat_ids`.\n"
            mem_blocks.append(block)
            
        prompt = NEW_MEMORY_FINALIZATION_PROMPT.format(
            game_rules=game_rules,
            new_questions="\n\n".join(mem_blocks),
            max_stats_per_memory=self.max_stats_per_memory
        )
        
        msg = [{"role": "user", "content": prompt}]
        resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
        
        self._log_prompt("PHASE D: NEW MEMORY FINALIZATION", prompt, resp[0])
        
        parsed = extract_json_block(strip_thinking_block(resp[0]))
        
        finalized = {}
        for upd in parsed.get("finalized_memories", []):
            qid_str = upd.get("question_id", "")
            match = re.search(r'\[Question\s+(\d+)\]', qid_str, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                finalized[idx] = upd
        
        for i, m_info in mem_map.items():
            draft = m_info["draft"]
            added = m_info["added"]
            
            q = draft.get("question")
            content = ""
            
            upd = finalized.get(i)
            evict_ids = []
            if upd:
                if upd.get("update") and upd.get("new_content"):
                    content = upd["new_content"]
                evict_ids = upd.get("evict_stat_ids") or []
                
            if not content:
                content = f"Active Tracker: {q}"
                
            final_stat_ids = list(dict.fromkeys([s for s in added if s not in evict_ids]))
                
            mem_id = f"mem_{uuid.uuid4().hex[:8]}"
            vec = self.embedder.encode(content, is_query=False)
            
            mem_id = self.store.add_memory(
                key=opponent_key,
                content=content,
                vec=vec,
                stat_ids=final_stat_ids,
                question=q
            )
            for sid in final_stat_ids:
                self.stat_pool.add_reference(opponent_key, sid, mem_id)

    def flush_batch_updates(self, gradient_data: list) -> None:
        """
        Main orchestration loop for post-game processing. 
        Replaces the old serial per-game execution with a batched pipeline:
        
        1. Phase A: Extract stat proposals individually for each game.
        2. Phase B: Batch all proposals across all games and resolve them semantically (inherit/define) in ONE LLM call.
        3. Phase 1 Loop: Update numerical values of all involved stats based on the game trajectory.
        4. Phase C: Batched update for textual content of existing memories affected by stat changes or new info requests.
        5. Phase D: Batched finalization for textual content of newly created memories.
        6. Strategy Merging (Legacy system).
        7. Persist the store and stat_pool to disk.
        """
        if not gradient_data:
            return
            
        # Group gradient data by opponent key
        grouped_data = {}
        for data in gradient_data:
            opp_key = data.get("opponent_key")
            if not opp_key:
                continue
            if opp_key not in grouped_data:
                grouped_data[opp_key] = []
            grouped_data[opp_key].append(data)
            
        for target_key, opp_gradient_data in grouped_data.items():
            all_new_strategies = []

            all_proposed_stats = []
            per_game_results = {}

            # 0. Score memories based on in-game utility (assessed during chat_step)
            for game_data in opp_gradient_data:
                opp_key = game_data.get("opponent_key")
                q_log = game_data.get("question_log", [])
                for q_entry in q_log:
                    d_id = q_entry.get("driving_memory_id")
                    src_id = q_entry.get("source_memory_id")
                    answered = q_entry.get("answered", False)

                    if d_id and answered:
                        self.store.update_score(opp_key, d_id, 1)

                    if src_id and not answered:
                        self.store.update_score(opp_key, src_id, -1)

            # 1. Phase A Loop (Concurrent)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(opp_gradient_data)) as executor:
                future_to_idx = {}
                for game_idx, game_data in enumerate(opp_gradient_data):
                    opponent_key = game_data.get("opponent_key")

                    q_log = game_data.get("question_log", [])
                    gh = game_data.get("game_history", "")

                    # Fetch game rules assigned by main.py
                    game_rules = getattr(self, 'current_game_intro', "")

                    if target_key and q_log and gh:
                        future = executor.submit(self._run_phase_a, target_key, q_log, gh, game_rules)
                        future_to_idx[future] = (game_idx, gh, game_rules)

                for future in concurrent.futures.as_completed(future_to_idx):
                    game_idx, gh, game_rules = future_to_idx[future]
                    res = future.result()
                    per_game_results[game_idx] = {
                        "stat_proposals": res.stat_proposals,
                        "gh": gh,
                        "game_rules": game_rules
                    }

            # 1.5 Split into new_questions vs desired_infos & Phase A.5 Merge
            all_raw_new_questions = []
            global_desired_infos = []
            for res in per_game_results.values():
                for p in res["stat_proposals"]:
                    q_text = p.get("question", "")
                    if q_text:
                        p["question"] = re.sub(r'(?i)\s*desired\s+(?:additional\s+)?info.*', '', q_text, flags=re.DOTALL).strip()
                    mem_id = p.get("memory_id")
                    if mem_id:
                        mem = self.store.get_memory(target_key, mem_id)
                        if mem:
                            global_desired_infos.append({
                                "memory_id": mem_id,
                                "desired_additional_info": p.get("desired_info", "Detailed investigation requested."),
                                "proposed_stats": p.get("proposed_stats", [])
                            })
                        else:
                            # Fallback: if memory_id is unmatched, treat as a new question
                            all_raw_new_questions.append(p)
                    else:
                        all_raw_new_questions.append(p)

            gr = list(per_game_results.values())[0]["game_rules"] if per_game_results else ""
            merged_new_questions = self._run_phase_a_5_merge(all_raw_new_questions, gr) if all_raw_new_questions else []

            all_proposed_stats = []
            for q in merged_new_questions:
                all_proposed_stats.extend(q.get("proposed_stats", []))
            for d in global_desired_infos:
                all_proposed_stats.extend(d.get("proposed_stats", []))

            # 2. Phase B
            resolved_stats_map = self._run_phase_b(target_key, all_proposed_stats, gr)
            # 3. Phase 1 Loop: Numerical Update (Concurrent)
            global_changed_stat_info = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(per_game_results))) as executor:
                future_to_res = {}
                for game_idx, res in per_game_results.items():
                    gh = res["gh"]
                    gr = res["game_rules"]

                    # Fetch stats only from memories that were retrieved/driven this round
                    retrieved_mem_ids = set()
                    for g_data in opp_gradient_data:
                        for q_entry in g_data.get("question_log", []):
                            if q_entry.get("driving_memory_id"):
                                retrieved_mem_ids.add(q_entry["driving_memory_id"])
                            if q_entry.get("source_memory_id"):
                                retrieved_mem_ids.add(q_entry["source_memory_id"])
                            for m_id in q_entry.get("retrieved_memory_ids", []):
                                retrieved_mem_ids.add(m_id)

                    existing_stat_ids = []
                    for m_id in retrieved_mem_ids:
                        mem = self.store.get_memory(target_key, m_id)
                        if mem:
                            existing_stat_ids.extend(mem.get("stat_ids", []))
                    existing_stat_ids = list(set(existing_stat_ids))

                    # Since all newly resolved stats from Phase B belong to the opponent,
                    # we pass ALL of them to every game's Phase 1 so they can be updated by all trajectories!
                    newly_resolved_ids = list(set(resolved_stats_map.values()))

                    # Run numerical update for this specific game concurrently
                    future = executor.submit(self._run_stat_update, target_key, gh, gr, existing_stat_ids, newly_resolved_ids)
                    future_to_res[future] = res

                for future in concurrent.futures.as_completed(future_to_res):
                    res = future_to_res[future]
                    changed_stat_info = future.result()

                    # Accumulate global changes for the batch
                    for m_id, s_ids in changed_stat_info.items():
                        if m_id not in global_changed_stat_info:
                            global_changed_stat_info[m_id] = []
                        global_changed_stat_info[m_id].extend(s_ids)

            global_new_mems = merged_new_questions

            # 4. Phase C: Batched Content Update
            if global_changed_stat_info or global_desired_infos:
                # Deduplicate stat IDs per memory
                for m_id in global_changed_stat_info:
                    global_changed_stat_info[m_id] = list(set(global_changed_stat_info[m_id]))

                self._run_memory_content_update(
                    target_key,
                    gr,
                    global_changed_stat_info,
                    global_desired_infos,
                    resolved_stats_map,
                    all_proposed_stats
                )

            # 5. Phase D: Batched New Memory Finalization
            if global_new_mems:
                self._run_new_memory_finalization(
                    target_key,
                    global_new_mems,
                    resolved_stats_map,
                    all_proposed_stats,
                    gr
                )

            # 5.5. GC: Remove orphaned Phase B stats that were never bound to any memory
            newly_resolved_ids = list(set(resolved_stats_map.values()))
            for sid in newly_resolved_ids:
                if target_key in self.stat_pool.stats and sid in self.stat_pool.stats[target_key]:
                    refs = self.stat_pool.stats[target_key][sid].get("referenced_by", [])
                    if len(refs) == 0:
                        del self.stat_pool.stats[target_key][sid]

            # 6. Legacy Strategy Merging (only when strategy memory is enabled)
            if self.use_strategy_memory:
                for game_data in opp_gradient_data:
                    s_log = game_data.get("strategy_log", [])
                    gh = game_data.get("game_history", "")
                    if s_log and gh:
                        new_s = self._process_strategy_log(s_log, gh)
                        if new_s:
                            all_new_strategies.extend(new_s)

                if all_new_strategies:
                    self._merge_new_strategies(all_new_strategies)

        # 7. Persist
        self.store.save(self.store_path)
        self.stat_pool.save(self.stat_pool_path)
        if self.use_strategy_memory:
            self.strategy_store.save(self.strategy_store_path)
