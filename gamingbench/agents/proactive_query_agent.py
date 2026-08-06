import os
import re
import json
import threading
import copy
from typing import List, Dict, Any

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.two_layer_store import TwoLayerStore
from gamingbench.ltm.strategy_store import StrategyStore
from gamingbench.ltm.two_layer_prompts import (
    PQA_QUESTION_GEN_PROMPT,
    PQA_MEMORY_MODIFY_PROMPT,
    PQA_UNANSWERED_SYNTHESIS_PROMPT,
    POST_GAME_QUESTION_REVIEW_PROMPT,
    ROUTE_AND_MODIFY_PROMPT,
    PROACTIVE_INJECTION_BLOCK,
    IN_GAME_ASSESSMENT_SUFFIX,
    STRATEGY_INJECTION_BLOCK,
    STRATEGY_SCORING_PROMPT,
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
        return json.loads(json_str)
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
        
        self.store_path = getattr(config, "two_layer_store_path", "two_layer_store.json")
        self.store = TwoLayerStore()
        
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)
            
        self.strategy_store_path = getattr(config, "strategy_store_path", "strategy_store.json")
        self.strategy_store = StrategyStore()
        if os.path.exists(self.strategy_store_path):
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

        strat_base = os.path.basename(self.strategy_store_path)
        self.strategy_store_path = os.path.join(storage_dir, strat_base)
        if os.path.exists(self.strategy_store_path):
            self.strategy_store.load(self.strategy_store_path)

    def reset_game_state(self, opponent_key, game_intro):
        if not self.batch_mode:
            if os.path.exists(self.store_path): 
                self.store.load(self.store_path)
            if os.path.exists(self.strategy_store_path):
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

    def _get_strategy_injection_block(self):
        top_strats = self.strategy_store.get_top_k_by_score(6)
        if top_strats:
            strat_lines = []
            for s in top_strats:
                strat_lines.append(f"[{s['id']}] \"{s['title']}\" (✓{s.get('success_count', 0)} / ~{s.get('neutral_count', 0)} / ✗{s.get('failure_count', 0)})\nDefinition: {s['definition']}\n")
            return STRATEGY_INJECTION_BLOCK.format(top_strategies="\n".join(strat_lines))
        else:
            return STRATEGY_INJECTION_BLOCK.format(top_strategies="No strategies available yet. You must create a new one.")

    def _generate_summary_and_question(self, observations):
        """
        Runs the pre-action LLM call to summarize the state AND ask a strategic question.
        """
        sys_prompt, obs_prompt = PromptAgent._build_prompts(self, observations)
        
        # Fetch top 10 memories by score for the prompt injection
        top_questions_text = "No prior top-performing questions available."
        if self.current_opponent_key:
            top_mems = self.store.get_top_k_by_score(self.current_opponent_key, top_k=10)
            if top_mems:
                top_q_list = []
                for m in top_mems:
                    score = m.get("score", 0)
                    top_q_list.append(f"[{m['id']}] \"{m['question']}\"")
                top_questions_text = "\n".join(top_q_list)
                
        strat_injection = self._get_strategy_injection_block()
                
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"{strat_injection}\n--- CURRENT GAME STATE ---\n{obs_prompt}\n\n{PQA_QUESTION_GEN_PROMPT.format(top_questions=top_questions_text, working_memory=self.match_working_memory, max_questions=self.max_questions_per_step)}"}
        ]
        
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        stripped_resp = strip_thinking_block(raw_resp)
        
        self.logger.info("=== IN-GAME QUESTION GENERATION ===")
        self.logger.info(f"PROMPT:\n{messages[1]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{stripped_resp}")
        self.logger.info("===================================")
        
        # Extract the JSON block from the LLM's response
        parsed_json = extract_json_block(stripped_resp)
        # Parse the summary text
        summary = parsed_json.get("summary", "")
        # Parse the questions array (defaults to an empty list if not found)
        questions_raw = parsed_json.get("questions", [])
        
        questions = []
        # Ensure the parsed questions field is actually a list
        if isinstance(questions_raw, list):
            # Iterate through each generated question object
            for q in questions_raw:
                # Ensure the question object is a dictionary
                if isinstance(q, dict):
                    # Extract the question text
                    q_text = q.get("question", "")
                    # Extract the source memory ID, if provided
                    src_id = q.get("source_memory_id")
                    # If it's a blank string, convert it to None for consistency
                    if isinstance(src_id, str) and not src_id.strip():
                        src_id = None
                    # Only append valid non-empty questions
                    if q_text:
                        # SAFE FALLBACK: If source_memory_id is provided, perfectly copy the original question text
                        if src_id:
                            original_mem = self.store.get_memory(self.current_opponent_key, src_id)
                            if original_mem and 'question' in original_mem:
                                q_text = original_mem['question']
                                
                        questions.append({"question": q_text, "source_memory_id": src_id})
        
        # Fallback in case the LLM fails to format as an array and returns the old single-question format
        if not questions:
            # Extract the single question text
            q_text = parsed_json.get("question", raw_resp.strip())
            # Extract the single source memory ID
            src_id = parsed_json.get("source_memory_id")
            # Convert blank string to None
            if isinstance(src_id, str) and not src_id.strip():
                src_id = None
            # Append as a 1-element list
            if q_text:
                if src_id:
                    original_mem = self.store.get_memory(self.current_opponent_key, src_id)
                    if original_mem and 'question' in original_mem:
                        q_text = original_mem['question']
                questions.append({"question": q_text, "source_memory_id": src_id})
        
        return summary, questions, query
        
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
                combined_query = f"Summary: {summary_text}\nQuestion: {question_text}"
                query_vec = self.embedder.encode(combined_query, is_query=True)
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
        """Main action loop for the proactive agent on its turn."""
        self.move_count += 1
        query_list = []
        
        summary_text, questions, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        # Log the current step for the post-game trajectory
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        
        # Build prompt injections for all questions by iteratively retrieving memories
        all_retrieved_mems_list = []
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
            # Store the retrieved memories to assess them later
            all_retrieved_mems_list.append((q_dict, retrieved_mems))
            
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        
        strat_injection = self._get_strategy_injection_block()
        if all_retrieved_mems_list:
            combined_injection = self._build_injection_block(all_retrieved_mems_list)
            observation_prompt = observation_prompt.replace(board_state, strat_injection + "\n\n" + combined_injection + "\n\n" + board_state, 1)
        else:
            observation_prompt = observation_prompt.replace(board_state, strat_injection + "\n\n" + board_state, 1)

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead."
            
        step_prompt += IN_GAME_ASSESSMENT_SUFFIX
            
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
                
                # Parse strategy
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
                    required = ["title", "definition", "success_criteria", "neutral_criteria", "failure_criteria"]
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
                            
                        q_type = item.get("question_type", "new")
                        
                        if q_type == "direct":
                            ans = True
                            mem_concl = ""
                            d_id = all_retrieved_mems_list[target_idx][0].get("source_memory_id")
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
            
        if 'parsed_strategy' not in locals() or parsed_strategy is None:
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
        """Custom chat step loop using proactive question generation."""
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        self.move_count += 1
        query_list = []
        observations['chat_context'] = chat_history_str
        
        summary_text, questions, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        
        # Build prompt injections for all questions by iteratively retrieving memories
        all_retrieved_mems_list = []
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
        
        strat_injection = self._get_strategy_injection_block()
        if all_retrieved_mems_list:
            combined_injection = self._build_injection_block(all_retrieved_mems_list)
            observation_prompt = observation_prompt.replace(board_state, strat_injection + "\n\n" + combined_injection + "\n\n" + board_state, 1)
        else:
            observation_prompt = observation_prompt.replace(board_state, strat_injection + "\n\n" + board_state, 1)

        if env_name == 'cooperative_negotiation':
            from gamingbench.prompts.chat_prompts import COOP_CHAT_INSTRUCTION as instruction
        else:
            from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION as instruction
            
        observation_prompt = observation_prompt + '\n\n' + instruction
        
        observation_prompt += IN_GAME_ASSESSMENT_SUFFIX
            
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
                
                # Parse strategy
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
                    required = ["title", "definition", "success_criteria", "neutral_criteria", "failure_criteria"]
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
                            
                        q_type = item.get("question_type", "new")
                        
                        if q_type == "direct":
                            ans = True
                            mem_concl = ""
                            d_id = all_retrieved_mems_list[target_idx][0].get("source_memory_id")
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
            
        if 'parsed_strategy' not in locals() or parsed_strategy is None:
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
        
        # Check if we have a specific player index (e.g., Player 1 or Player 2).
        player_index = getattr(self, 'current_player_index', None)
        if player_index is not None:
            # Replace the generic "Player X" with "You" in the game trajectory so the LLM knows which perspective it is evaluating from.
            game_history = game_history.replace(f"Player {player_index}", "You")

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
            # Write the updated memory bank back to disk so it's ready for the next game.
            self.store.save(self.store_path)
            
        new_s = self._process_strategy_log(strategy_log_data, game_history)
        if new_s:
            self._merge_new_strategies(new_s)
            
        if self.strategy_store_path != '/dev/null':
            self.strategy_store.save(self.strategy_store_path)

    def _process_strategy_log(self, strategy_log, game_history):
        """Scores followed strategies and merges/scores new strategies at the end of a game/batch."""
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
            
        # 2. Score strategies (followed and new)
        scores = self._score_strategies(followed_strats, new_strats, game_history)
        
        # 3. Apply scores to followed strategies
        for s_id, score in scores.items():
            if s_id.startswith("strat_"):
                self.strategy_store.update_score(s_id, score)
                
        # 4. Return new strategies with their scores for batch-level merging
        new_strats_with_scores = []
        if new_strats:
            for idx, s in enumerate(new_strats):
                temp_id = f"temp_new_{idx}"
                new_strats_with_scores.append((s, scores.get(temp_id, "neutral")))
                
        return new_strats_with_scores

    def _score_strategies(self, followed_strats, new_strats, game_history):
        strategies_to_score_text = ""
        for s in followed_strats:
            strategies_to_score_text += f"ID: {s['id']}\nTitle: {s['title']}\nDefinition: {s['definition']}\nSuccess Criteria: {s['success_criteria']}\nNeutral Criteria: {s['neutral_criteria']}\nFailure Criteria: {s['failure_criteria']}\n\n"
            
        for idx, s in enumerate(new_strats):
            temp_id = f"temp_new_{idx}"
            strategies_to_score_text += f"ID: {temp_id}\nTitle: {s['title']}\nDefinition: {s['definition']}\nSuccess Criteria: {s['success_criteria']}\nNeutral Criteria: {s['neutral_criteria']}\nFailure Criteria: {s['failure_criteria']}\n\n"
            
        messages = [
            {'role': 'system', 'content': "You are an expert game strategist evaluating the performance of game strategies."},
            {'role': 'user', 'content': STRATEGY_SCORING_PROMPT.format(strategies_to_score=strategies_to_score_text.strip(), game_trajectory=game_history)}
        ]
        
        scores_map = {}
        max_retries = 3
        for attempt in range(max_retries):
            responses, _ = self.llm_query(messages, n=1, stop=None, prompt_type='move')
            
            if attempt == 0:
                self.logger.info(f'Strategy Scoring Prompt: {messages[1]["content"]}')
            self.logger.info(f'Strategy Scoring Raw Response (Attempt {attempt+1}): {responses}')
            
            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Strategy Scoring Stripped Response (Attempt {attempt+1}):\n{stripped_response}')
            try:
                parsed_json = extract_json_block(stripped_response)
                
                log_dir = os.path.dirname(self.strategy_store_path)
                if log_dir and log_dir != '/dev/null':
                    s_log_file = os.path.join(log_dir, f"{self.agent_name}_strategy_processing.log")
                    try:
                        with open(s_log_file, "a") as f:
                            f.write("=== STRATEGY SCORING ===\n")
                            f.write(f"PROMPT:\n{messages[1]['content']}\n")
                            f.write(f"RAW ANSWER:\n{raw_response}\n")
                            f.write(f"STRIPPED ANSWER:\n{stripped_response}\n")
                            f.write("========================\n\n")
                    except Exception as e:
                        self.logger.error(f"Failed to write to strategy processing log: {e}")
                        
                scores = parsed_json.get("scores", [])
                for s in scores:
                    s_id = s.get("strategy_id")
                    score_val = s.get("score", "").lower()
                    if s_id and score_val in ["success", "neutral", "failure"]:
                        scores_map[s_id] = score_val
                break
            except Exception as e:
                self.logger.warning(f"Failed to parse strategy scores on attempt {attempt+1}: {str(e)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": "Failed to parse JSON. Please try again with valid JSON format."})
                
        return scores_map

    def _merge_new_strategies(self, new_strats_with_scores):
        if not new_strats_with_scores:
            return
            
        new_strats_text = ""
        for idx, (s, score) in enumerate(new_strats_with_scores):
            temp_id = f"temp_new_{idx}"
            new_strats_text += f"ID: {temp_id}\nTitle: {s['title']}\nScore: {score}\nDefinition: {s['definition']}\n\n"
            
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
                
                for k in parsed_json.get("keep", []):
                    kept_ids.add(k)
                    
                for g in parsed_json.get("merge_groups", []):
                    if g.get("keep"):
                        kept_ids.add(g["keep"])
                break
            except Exception as e:
                self.logger.warning(f"Failed to parse strategy merge on attempt {attempt+1}: {str(e)}")
                messages.append({"role": "assistant", "content": raw_response})
                messages.append({"role": "user", "content": "Failed to parse JSON. Please try again with valid JSON format."})
                
        for idx, (s, score) in enumerate(new_strats_with_scores):
            temp_id = f"temp_new_{idx}"
            if temp_id in kept_ids or not kept_ids:
                new_id = self.strategy_store.add_strategy(
                    title=s["title"],
                    definition=s["definition"],
                    success_criteria=s["success_criteria"],
                    neutral_criteria=s["neutral_criteria"],
                    failure_criteria=s["failure_criteria"]
                )
                self.strategy_store.update_score(new_id, score)
            
    def _process_question_log(self, opponent_key, question_log, game_history):
        """Processes each question asked during the game to verify answers and patch memory gaps."""
        if not question_log:
            return [], []
            
        # Deduplicate question log entries based on source_memory_id.
        # If the agent asked for the exact same memory multiple times in one game, we only evaluate it once.
        deduped_log = []
        seen_sources_map = {}
        for q_entry in question_log:
            src_id = q_entry.get("source_memory_id")
            if src_id:
                # Skip this question if we have already seen its source memory ID
                if src_id in seen_sources_map:
                    first_entry = seen_sources_map[src_id]
                    extra = q_entry.get("desired_additional_info", "")
                    if extra:
                        first_entry.setdefault("upgrade_requests", []).append(extra)
                    continue
                seen_sources_map[src_id] = q_entry
            # Add novel questions and unique direct memory fetches to the deduplicated log
            deduped_log.append(q_entry)
            
        section_a_blocks = []
        section_b_blocks = []
        question_map = {}
        for idx, q_entry in enumerate(deduped_log):
            q_id = f"Q{idx+1}"
            question_map[q_id] = q_entry
            
            is_direct = q_entry.get("is_direct_retrieval", False)
            
            if is_direct:
                formatted_q = f"[DIRECT] Question ID: {q_id}\n"
                formatted_q += f"Question: \"{q_entry['question']}\"\n"
                
                existing_mem_content = None
                lookup_id = q_entry.get("source_memory_id")
                if lookup_id:
                    existing_mem = self.store.get_memory(opponent_key, lookup_id)
                    if existing_mem:
                        existing_mem_content = existing_mem.get("content", "")
                if existing_mem_content:
                    formatted_q += f"CURRENT MEMORY CONTENT: \"{existing_mem_content}\"\n"
                    
                if q_entry.get('desired_additional_info'):
                    formatted_q += f"DESIRED ADDITIONAL INFO: {q_entry['desired_additional_info']}\n"
                if q_entry.get('upgrade_requests'):
                    formatted_q += f"UPGRADE REQUESTS:\n" + "\n".join([f"- {req}" for req in q_entry['upgrade_requests']]) + "\n"
                    
                section_a_blocks.append(formatted_q)
            else:
                formatted_q = f"[NEW] Question ID: {q_id}\n"
                formatted_q += f"Question: \"{q_entry['question']}\"\n"
                formatted_q += f"ANSWERED IN-GAME: {q_entry['answered']}\n"
                if 'memory_conclusion' in q_entry:
                    formatted_q += f"MEMORY CONCLUSION: {q_entry['memory_conclusion']}\n"
                if q_entry.get('desired_additional_info'):
                    formatted_q += f"DESIRED ADDITIONAL INFO: {q_entry['desired_additional_info']}\n"
                if q_entry.get('upgrade_requests'):
                    formatted_q += f"UPGRADE REQUESTS:\n" + "\n".join([f"- {req}" for req in q_entry['upgrade_requests']]) + "\n"
                
                driving_mem_content = None
                driving_id = q_entry.get("driving_memory_id")
                if driving_id:
                    driving_mem = self.store.get_memory(opponent_key, driving_id)
                    if driving_mem:
                        driving_mem_content = driving_mem.get("content", "")
                if driving_mem_content:
                    formatted_q += f"DRIVING MEMORY CONTENT: \"{driving_mem_content}\"\n"
                
                section_b_blocks.append(formatted_q)
                
        question_log_text = ""
        if section_a_blocks:
            question_log_text += "=== SECTION A (Direct Retrieval Questions) ===\n\n"
            question_log_text += "\n---\n".join(section_a_blocks)
            question_log_text += "\n\n"
        if section_b_blocks:
            question_log_text += "=== SECTION B (New Questions) ===\n\n"
            question_log_text += "\n---\n".join(section_b_blocks)
            question_log_text += "\n\n"
            
        question_log_text = question_log_text.strip()
        
        prompt = POST_GAME_QUESTION_REVIEW_PROMPT.format(
            question_log=question_log_text,
            game_trajectory=game_history
        )
        
        responses, query = self.llm_query([{"role": "user", "content": prompt}], n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        stripped_resp = strip_thinking_block(raw_resp)
        
        self.logger.info("=== QUESTION REVIEW ===")
        self.logger.info(f"PROMPT:\n{prompt}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{stripped_resp}")
        self.logger.info("=======================")
        
        log_dir = os.path.dirname(self.store_path)
        if log_dir and log_dir != '/dev/null':
            q_log_file = os.path.join(log_dir, f"{self.agent_name}_question_processing.log")
            try:
                with open(q_log_file, "a") as f:
                    f.write("=== QUESTION REVIEW ===\n")
                    f.write(f"PROMPT:\n{prompt}\n")
                    f.write(f"RAW ANSWER:\n{raw_resp}\n")
                    f.write(f"STRIPPED ANSWER:\n{stripped_resp}\n")
                    f.write("=======================\n\n")
            except Exception as e:
                self.logger.error(f"Failed to write to question processing log: {e}")
            
        parsed_json = extract_json_block(stripped_resp)
        
        reviews = parsed_json.get("question_reviews", [])
        if not isinstance(reviews, list):
            self.logger.error(f"Failed to parse question_reviews as list: {reviews}")
            return
            
        all_new_evidence = []
        correction_tasks_data = []
        synthesis_tasks_data = []
        
        for review in reviews:
            if not isinstance(review, dict):
                continue
                
            q_id = review.get("question_id")
            if q_id not in question_map:
                continue
                
            q_entry = question_map[q_id]
            tag = review.get("tag", "").upper()
            evidence_list = review.get("evidence", [])
            
            if q_entry.get("is_direct_retrieval"):
                if tag == "UNANSWERED":
                    self.logger.warning("[ROUTING GUARD] UNANSWERED tag on a direct-retrieval question. Force-rerouting to MODIFY.")
                    tag = "MODIFY"
            else:
                if tag == "UNANSWERED" and q_entry.get("driving_memory_id"):
                    self.logger.warning("[ROUTING GUARD] UNANSWERED tag on a new question that has a driving_memory_id. Force-rerouting to MODIFY.")
                    tag = "MODIFY"
            if tag == "REINFORCE":
                # PATH 1: REINFORCE
                for m_id in q_entry.get("retrieved_memory_ids", []):
                    self.store.update_score(opponent_key, m_id, 1)
                
                if isinstance(evidence_list, list) and len(evidence_list) > 0:
                    for ev in evidence_list:
                        if isinstance(ev, dict):
                            ev_content = ev.get("content", "")
                            db_vec = self.embedder.encode(ev_content, is_query=False).tolist()
                            ev_id = self.store.add_evidence(
                                key=opponent_key,
                                content=ev_content,
                                observation=f"Supporting evidence for question: '{q_entry['question']}'. Original observation: {ev.get('observation', '')}",
                                game_id=f"game_{getattr(self, 'game_count', 0)}",
                                vec=db_vec
                            )
                            for m_id in q_entry.get("retrieved_memory_ids", []):
                                mem = self.store.get_memory(opponent_key, m_id)
                                if mem and ev_id not in mem.get('evidence_ids', []):
                                    mem.setdefault('evidence_ids', []).append(ev_id)
                                    mem['evidence_ids'] = mem['evidence_ids'][-self.max_evidence_per_memory:]
                                    
            elif tag == "MODIFY":
                # PATH 2: MODIFY
                culprit_id = q_entry.get("driving_memory_id")
                if not culprit_id:
                    ret_mems = q_entry.get("retrieved_memory_ids", [])
                    culprit_id = q_entry.get("source_memory_id") or (ret_mems[0] if ret_mems else None)
                    
                if culprit_id:
                    self.store.update_score(opponent_key, culprit_id, -1)
                    current_mem = self.store.get_memory(opponent_key, culprit_id)
                    
                    if current_mem and isinstance(evidence_list, list) and len(evidence_list) > 0:
                        desired_info = q_entry.get("desired_additional_info", "")
                        upgrade_reqs = q_entry.get("upgrade_requests", [])
                        if upgrade_reqs:
                            desired_info = " / ".join(upgrade_reqs)
                            
                        ev_prefix = "Corrective and Enrichment" if desired_info else "Corrective"
                        
                        new_ev_blocks = []
                        new_ev_ids = []
                        for item in evidence_list:
                            if isinstance(item, dict):
                                ev_content = item.get("content", "")
                                db_vec = self.embedder.encode(ev_content, is_query=False).tolist()
                                obs_text = f"{ev_prefix} evidence for question: '{q_entry['question']}'."
                                if desired_info:
                                    obs_text += f" Desired enrichment: '{desired_info}'."
                                obs_text += f" Original observation: {item.get('observation', '')}"
                                
                                ev_id = self.store.add_evidence(
                                    key=opponent_key,
                                    content=ev_content,
                                    observation=obs_text,
                                    game_id=f"game_{getattr(self, 'game_count', 0)}",
                                    vec=db_vec
                                )
                                new_ev_ids.append(ev_id)
                                new_ev_blocks.append(f"EVIDENCE ID: {ev_id}\nCONTENT: {item.get('content')}\nOBSERVATION: {item.get('observation')}")
                        
                        correcting_evidence_str = "\n\n---\n\n".join(new_ev_blocks)
                        
                        old_ev_blocks = []
                        for eid in current_mem.get('evidence_ids', []):
                            ev = self.store.get_evidence(opponent_key, eid)
                            if ev:
                                old_ev_blocks.append(f"EVIDENCE ID: {eid}\nCONTENT: {ev.get('content')}")
                        supporting_evidence_str = "\n\n---\n\n".join(old_ev_blocks) if old_ev_blocks else "No supporting historical evidence."
                        
                        correction_tasks_data.append({
                            "memory_id": culprit_id,
                            "question": q_entry['question'],
                            "current_memory_content": current_mem['content'],
                            "supporting_evidence": supporting_evidence_str,
                            "correcting_evidence": correcting_evidence_str,
                            "new_ev_ids": new_ev_ids,
                            "desired_additional_info": desired_info
                        })
            elif tag == "UNANSWERED":
                # PATH 3: UNANSWERED
                if isinstance(evidence_list, list) and len(evidence_list) > 0:
                    new_ev_blocks = []
                    new_ev_ids = []
                    for item in evidence_list:
                        if isinstance(item, dict):
                            ev_content = item.get("content", "")
                            db_vec = self.embedder.encode(ev_content, is_query=False).tolist()
                            ev_id = self.store.add_evidence(
                                key=opponent_key,
                                content=ev_content,
                                observation=f"Answer to previously unanswered question: '{q_entry['question']}'. Original observation: {item.get('observation', '')}",
                                game_id=f"game_{getattr(self, 'game_count', 0)}",
                                vec=db_vec
                            )
                            new_ev_ids.append(ev_id)
                            new_ev_blocks.append(f"EVIDENCE ID: {ev_id}\nCONTENT: {item.get('content')}\nOBSERVATION: {item.get('observation')}")
                    
                    new_evidence_str = "\n\n---\n\n".join(new_ev_blocks)
                    
                    desired_info = q_entry.get("desired_additional_info", "")
                    upgrade_reqs = q_entry.get("upgrade_requests", [])
                    if upgrade_reqs:
                        desired_info = " / ".join(upgrade_reqs)
                        
                    synthesis_tasks_data.append({
                        "question": q_entry['question'],
                        "new_evidence_list": new_evidence_str,
                        "new_ev_ids": new_ev_ids,
                        "desired_additional_info": desired_info
                    })

        return correction_tasks_data, synthesis_tasks_data

    def _execute_batched_corrections(self, opponent_key, correction_tasks_data):
        """Processes all accumulated memory modifications (Path 2) across a batch into a single LLM call."""
        # 1. Early exit if there are no correction tasks.
        if correction_tasks_data:
            formatted_tasks = []
            ev_id_map = {}
            
            # 2. Group tasks by memory_id to avoid prompt duplication and evidence overwrite bugs
            grouped_tasks = {}
            for task in correction_tasks_data:
                m_id = task["memory_id"]
                if m_id not in grouped_tasks:
                    grouped_tasks[m_id] = {
                        "memory_id": m_id,
                        "questions": set(),
                        "current_memory_content": task["current_memory_content"],
                        "supporting_evidence": task["supporting_evidence"],
                        "correcting_evidence": [],
                        "new_ev_ids": [],
                        "desired_additional_info": set()
                    }
                grouped_tasks[m_id]["questions"].add(task["question"])
                if task["correcting_evidence"]:
                    grouped_tasks[m_id]["correcting_evidence"].append(task["correcting_evidence"])
                if task["new_ev_ids"]:
                    grouped_tasks[m_id]["new_ev_ids"].extend(task["new_ev_ids"])
                if task.get("desired_additional_info"):
                    grouped_tasks[m_id]["desired_additional_info"].add(task["desired_additional_info"])
            
            # 3. Format the grouped tasks
            for m_id, task in grouped_tasks.items():
                # Dedup the new evidence IDs and map them to the memory
                ev_id_map[m_id] = list(set(task["new_ev_ids"]))
                
                questions_str = " / ".join(task["questions"])
                corr_ev_str = "\n\n---\n\n".join(task["correcting_evidence"]) if task["correcting_evidence"] else "No correcting evidence provided."
                
                # 4. Format the task block with the grouped context
                task_str = f"--- TASK ID: {m_id} ---\nQUESTIONS IT ANSWERS: \"{questions_str}\"\nCURRENT MEMORY CONTENT: \"{task['current_memory_content']}\"\nSUPPORTING HISTORICAL EVIDENCE:\n{task['supporting_evidence']}\nNEW EVIDENCE FOR MODIFICATION:\n{corr_ev_str}"
                
                if task["desired_additional_info"]:
                    desired_info_str = " / ".join(task["desired_additional_info"])
                    task_str += f"\nDESIRED ADDITIONAL INFO (Agent Request): \"{desired_info_str}\""
                    
                formatted_tasks.append(task_str)
            
            # 5. Join all formatted tasks into a single large prompt string.
            correction_tasks_str = "\n\n".join(formatted_tasks)
            # 6. Inject the tasks into the correction prompt template.
            from gamingbench.ltm.two_layer_prompts import PQA_MEMORY_MODIFY_PROMPT
            correction_prompt = PQA_MEMORY_MODIFY_PROMPT.format(modification_tasks=correction_tasks_str)
            
            # 7. Query the LLM to rewrite all faulty memories at once.
            msg = [{"role": "user", "content": correction_prompt}]
            resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
            
            # 8. Log the input and output to standard logging.
            self.logger.info("=== TARGETED MEMORY CORRECTION (BATCHED) ===")
            self.logger.info(f"PROMPT:\n{correction_prompt}")
            self.logger.info(f"RAW ANSWER:\n{resp[0]}")
            self.logger.info("============================================")
            
            # 9. Also append this transaction to the dedicated question_processing log file.
            log_dir = os.path.dirname(self.store_path)
            if log_dir and log_dir != '/dev/null':
                q_log_file = os.path.join(log_dir, f"{self.agent_name}_question_processing.log")
                try:
                    with open(q_log_file, "a") as f:
                        f.write("=== TARGETED MEMORY CORRECTION (BATCHED) ===\n")
                        f.write(f"PROMPT:\n{correction_prompt}\n")
                        f.write(f"RAW ANSWER:\n{resp[0]}\n")
                        f.write("============================================\n\n")
                except Exception as e:
                    self.logger.error(f"Failed to write correction to question processing log: {e}")
            
            # 10. Extract and parse the JSON returned by the LLM.
            parsed = extract_json_block(strip_thinking_block(resp[0]))
            corrections = parsed.get("modifications", [])
            
            # 11. Iterate through the LLM's suggested memory rewrites.
            if isinstance(corrections, list):
                for corr in corrections:
                    if not isinstance(corr, dict): continue
                    m_id = corr.get("memory_id")
                    new_content = corr.get("memory_content")
                    
                    # 12. If a valid memory ID and rewritten content are provided...
                    if m_id and new_content:
                        # 13. Create a new vector embedding for the updated text.
                        vec = self.embedder.encode(new_content, is_query=False)
                        # 14. Update the memory in the database, merging the old content with the new.
                        # It pulls the corresponding new evidence IDs from the ev_id_map we created earlier.
                        # NOTE: we intentionally do NOT pass a `question` parameter here — the question
                        # field of a memory is immutable and must never be overwritten by the LLM output.
                        self.store.update_memory(
                            key=opponent_key,
                            memory_id=m_id,
                            new_content=new_content,
                            new_evidence_ids=ev_id_map.get(m_id, []),
                            vec=vec,
                            max_evidence_per_memory=self.max_evidence_per_memory
                        )

    def _execute_batched_synthesis(self, opponent_key, synthesis_tasks_data):
        """Processes all unanswered questions (Path 3) across a batch to synthesize brand new memories."""
        # 1. Early exit if there are no unanswered synthesis tasks.
        if synthesis_tasks_data:
            formatted_tasks = []
            all_new_ev_ids = set()
            relevant_historical_ev_ids = set()
            
            # 2. Iterate through each unanswered question to gather relevant background evidence.
            for task in synthesis_tasks_data:
                # Keep track of all new evidence IDs generated so we don't accidentally retrieve them as "historical" evidence below.
                all_new_ev_ids.update(task["new_ev_ids"])
                
                # 3. We want to supply the LLM with older, historical facts that might help answer the question.
                # We combine the question and the new evidence into a single query vector.
                combined_query = f"Question: {task['question']}\nNew Evidence:\n{task['new_evidence_list']}"
                query_vec = self.embedder.encode(combined_query, is_query=True)
                
                # 4. Perform a semantic search on the evidence database (Layer 1) to find the top 3 most relevant historical facts.
                top_ev = self.store.find_relevant_evidence(opponent_key, query_vec, top_k=3)
                for ev in top_ev:
                    relevant_historical_ev_ids.add(ev["id"])
                
                # 5. Format the question and the new evidence into a task block.
                task_str = f"--- QUESTION: \"{task['question']}\" ---\nNEW EVIDENCE:\n{task['new_evidence_list']}"
                if task.get("desired_additional_info"):
                    task_str += f"\nDESIRED ADDITIONAL INFO (Agent Request): \"{task['desired_additional_info']}\""
                formatted_tasks.append(task_str)
            
            # 6. Join all tasks into a single string to inject into the prompt.
            synthesis_tasks_str = "\n\n".join(formatted_tasks)
            
            # 7. Collect the actual content strings for the historical evidence IDs we found during semantic search.
            all_ev = self.store.get_all_evidence(opponent_key)
            related_ev_blocks = []
            for ev in all_ev:
                # Ensure we only include historical evidence (and exclude the brand new evidence we just generated this batch).
                if ev["id"] in relevant_historical_ev_ids and ev["id"] not in all_new_ev_ids:
                    related_ev_blocks.append(f"EVIDENCE ID: {ev['id']}\nCONTENT: {ev['content']}")
            
            # 8. Combine the historical evidence blocks into a string.
            related_evidence_str = "\n\n---\n\n".join(related_ev_blocks) if related_ev_blocks else "No related historical evidence."
            
            # 9. Format the synthesis prompt with the tasks and the supplementary historical evidence.
            synthesis_prompt = PQA_UNANSWERED_SYNTHESIS_PROMPT.format(
                synthesis_tasks=synthesis_tasks_str,
                related_evidence_list=related_evidence_str
            )
            
            # 10. Query the LLM to synthesize entirely new memories from these unanswered questions.
            msg = [{"role": "user", "content": synthesis_prompt}]
            resp, qry = self.llm_query(msg, n=1, stop=None, prompt_type='move')
            
            # 11. Log to standard console.
            self.logger.info("=== UNANSWERED QUESTION SYNTHESIS (BATCHED) ===")
            self.logger.info(f"PROMPT:\n{synthesis_prompt}")
            self.logger.info(f"RAW ANSWER:\n{resp[0]}")
            self.logger.info("===============================================")
            
            # 12. Log to the dedicated question_processing log file.
            log_dir = os.path.dirname(self.store_path)
            if log_dir and log_dir != '/dev/null':
                q_log_file = os.path.join(log_dir, f"{self.agent_name}_question_processing.log")
                try:
                    with open(q_log_file, "a") as f:
                        f.write("=== UNANSWERED QUESTION SYNTHESIS (BATCHED) ===\n")
                        f.write(f"PROMPT:\n{synthesis_prompt}\n")
                        f.write(f"RAW ANSWER:\n{resp[0]}\n")
                        f.write("===============================================\n\n")
                except Exception as e:
                    self.logger.error(f"Failed to write synthesis to question processing log: {e}")
            
            # 13. Parse the JSON response to extract the new memory objects.
            parsed = extract_json_block(strip_thinking_block(resp[0]))
            new_memories = parsed.get("new_memories", [])
            
            # 14. Iterate through the newly synthesized memories.
            if isinstance(new_memories, list):
                for mem in new_memories:
                    if not isinstance(mem, dict): continue
                    q = mem.get("question")
                    new_content = mem.get("memory_content")
                    used_ev_ids = mem.get("evidence_ids_used", [])
                    if not isinstance(used_ev_ids, list):
                        used_ev_ids = []
                    
                    # 15. If a valid question and memory content exist...
                    if q and new_content:
                        # 16. Embed the new memory for semantic storage.
                        vec = self.embedder.encode(new_content, is_query=False)
                        # 17. Add the brand new memory to the Layer 2 database, attaching the specific evidence IDs it used.
                        self.store.add_memory(
                            key=opponent_key,
                            content=new_content,
                            evidence_ids=used_ev_ids,
                            vec=vec,
                            max_evidence_per_memory=self.max_evidence_per_memory,
                            question=q
                        )

    def flush_batch_updates(self, gradient_data: list) -> None:
        """Called at the end of a parallel batch to accumulate all games and execute LLM updates across the batch."""
        # 1. Early exit if no data was returned from the batched games.
        if not gradient_data:
            return
            
        target_key = None
        all_correction_tasks = []
        all_synthesis_tasks = []
        all_new_strategies = []
        
        # 2. Iterate through the results of each game played in this parallel batch.
        for data in gradient_data:
            if isinstance(data, dict):
                target_key = data.get("opponent_key", target_key)
                q_log = data.get("question_log", [])
                s_log = data.get("strategy_log", [])
                gh = data.get("game_history", "")
                
                # 3. If the game data is valid, parse the questions (this runs the Review Prompt per-game internally).
                if target_key and q_log and gh:
                    # _process_question_log does the game-specific review and returns the structured tasks.
                    c_tasks, s_tasks = self._process_question_log(target_key, q_log, gh)
                    # 4. Accumulate these tasks into global batch lists.
                    all_correction_tasks.extend(c_tasks)
                    all_synthesis_tasks.extend(s_tasks)
                    
                if s_log and gh:
                    new_s = self._process_strategy_log(s_log, gh)
                    if new_s:
                        all_new_strategies.extend(new_s)
                    
        # 5. After all games in the batch have been reviewed and their tasks pooled...
        if target_key:
            # 6. Execute one giant LLM call to correct ALL flawed memories discovered in this batch.
            if all_correction_tasks:
                self._execute_batched_corrections(target_key, all_correction_tasks)
            # 7. Execute one giant LLM call to synthesize memories for ALL unanswered questions in this batch.
            if all_synthesis_tasks:
                self._execute_batched_synthesis(target_key, all_synthesis_tasks)
                
        if all_new_strategies:
            self._merge_new_strategies(all_new_strategies)
                
        # 8. Finally, save the updated two-layer database to disk so it's ready for the next batch/epoch.
        if target_key and self.store_path != '/dev/null':
            self.store.save(self.store_path)
            
        if self.strategy_store_path != '/dev/null':
            self.strategy_store.save(self.strategy_store_path)
