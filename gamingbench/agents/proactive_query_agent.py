import os
import re
import json
import threading
import copy
from typing import List, Dict, Any

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.two_layer_store import TwoLayerStore
from gamingbench.ltm.two_layer_prompts import (
    QUESTION_GEN_PROMPT,
    ANSWER_ASSESSMENT_PROMPT,
    POST_GAME_QUESTION_REVIEW_PROMPT,
    ROUTE_AND_MODIFY_PROMPT,
    PROACTIVE_INJECTION_PROMPT
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
        
        self.store_path = getattr(config, "two_layer_store_path", "two_layer_store.json")
        self.store = TwoLayerStore()
        
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)
            
        self.current_opponent_key = None
        self.current_game_intro = None
        self.move_count = 0
        self.current_trajectory = []
        self._last_batch_result = None
        
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

    def reset_game_state(self, opponent_key, game_intro):
        if not self.batch_mode:
            if os.path.exists(self.store_path): 
                self.store.load(self.store_path)
            
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

    def _generate_summary_and_question(self, observations):
        """
        Runs the pre-action LLM call to summarize the state AND ask a strategic question.
        """
        sys_prompt, obs_prompt = PromptAgent._build_prompts(self, observations)
        
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"--- CURRENT GAME STATE ---\n{obs_prompt}\n\n{QUESTION_GEN_PROMPT}"}
        ]
        
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        stripped_resp = strip_thinking_block(raw_resp)
        
        self.logger.info("=== IN-GAME QUESTION GENERATION ===")
        self.logger.info(f"PROMPT:\n{messages[1]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{stripped_resp}")
        self.logger.info("===================================")
        
        parsed_json = extract_json_block(stripped_resp)
        summary = parsed_json.get("summary", "")
        question = parsed_json.get("question", raw_resp.strip())
        
        return summary, question, query
        
    def _run_in_game_memory_retrieval(self, observations, summary_text, question_text, env_name, observation_prompt):
        """Helper to process the retrieval using both summary and question."""
        retrieved_mems = []
        
        if self.current_opponent_key and question_text:
            combined_query = f"Summary: {summary_text}\nQuestion: {question_text}"
            query_vec = self.embedder.encode(combined_query, is_query=True)
            retrieved_mems = self.store.find_relevant_memories(self.current_opponent_key, query_vec, top_k=self.in_game_top_k)
            
            if retrieved_mems:
                mem_texts = [m['content'] for m in retrieved_mems]
                text_blob = "\n\n---\n\n".join(mem_texts)
            else:
                text_blob = "None."
                
            injection = PROACTIVE_INJECTION_PROMPT.format(question=question_text, retrieved_memories=text_blob)
            
            board_state = construct_observation_prompt(observations, env_name)
            observation_prompt = observation_prompt.replace(board_state, injection + "\n\n" + board_state, 1)
                
        return retrieved_mems, observation_prompt
        
    def _assess_answer(self, retrieved_mems, question_text):
        """Helper to perform the answer assessment after retrieval."""
        answered = False
        memory_conclusion = ""
        query = None
        if retrieved_mems:
            mem_text = "\n".join([f"- {m['content']}" for m in retrieved_mems])
            assess_msg = [
                {"role": "user", "content": ANSWER_ASSESSMENT_PROMPT.format(question=question_text, retrieved_memories=mem_text)}
            ]
            assess_resp, assess_query = self.llm_query(assess_msg, n=1, stop=None, prompt_type='move')
            query = assess_query
            raw_assess_resp = assess_resp[0]
            stripped_assess_resp = strip_thinking_block(raw_assess_resp)
            
            self.logger.info("=== IN-GAME ANSWER ASSESSMENT ===")
            self.logger.info(f"PROMPT:\n{assess_msg[0]['content']}")
            self.logger.info(f"RAW ANSWER:\n{raw_assess_resp}")
            self.logger.info(f"STRIPPED ANSWER:\n{stripped_assess_resp}")
            self.logger.info("=================================")
            
            parsed_json = extract_json_block(stripped_assess_resp)
            ans = parsed_json.get("answered", False)
            if isinstance(ans, str):
                ans = ans.lower() == 'true' or ans.lower() == 'yes'
            answered = ans
            memory_conclusion = parsed_json.get("memory_conclusion", stripped_assess_resp.strip())
            
        return answered, memory_conclusion, query

    def step(self, observations):
        """Main action loop for the proactive agent on its turn."""
        self.move_count += 1
        query_list = []
        
        summary_text, question_text, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        # Log the current step for the post-game trajectory
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        
        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Action",
            "state": board_state,
            "summary": summary_text,
            "question": question_text
        })
        
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        retrieved_mems, observation_prompt = self._run_in_game_memory_retrieval(observations, summary_text, question_text, env_name, observation_prompt)

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead."
            
        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""
        
        for attempt in range(max_retries):
            responses, query = self.llm_query(msgs, n=self.num_generations, stop=None, prompt_type='move')
            query_list.append(query)
            
            if attempt == 0:
                self.logger.info(f'Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')
            
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
                    break
                else:
                    error_msg = f"Invalid move '{move}'. Your move must be one of the legal actions: {valid_moves}. Please try again."
            else:
                move = ""
                error_msg = f"Failed to extract a valid move format. Legal actions: {valid_moves}. Please try again."
                
            if attempt < max_retries - 1:
                msgs.append({"role": "assistant", "content": responses[0]})
                msgs.append({"role": "user", "content": error_msg})
                
        self.current_trajectory[-1]["action"] = f"[Move] {move}"
        
        answered, memory_conclusion, assess_query = self._assess_answer(retrieved_mems, question_text)
        if assess_query:
            query_list.append(assess_query)
            
        mem_text = "\n".join([f"- {m['content']}" for m in retrieved_mems]) if retrieved_mems else ""
        self.question_log.append({
            "round": observations.get('game_round', self.move_count),
            "question": question_text,
            "retrieved_memory_ids": [m["id"] for m in retrieved_mems],
            "retrieved_memories_text": mem_text if retrieved_mems else "No relevant memories found.",
            "answered": answered,
            "memory_conclusion": memory_conclusion
        })
        
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        """Custom chat step loop using proactive question generation."""
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        self.move_count += 1
        query_list = []
        observations['chat_context'] = chat_history_str
        
        # Generate summary AND proactive question just like in step()
        summary_text, question_text, sum_query = self._generate_summary_and_question(observations)
        if sum_query:
            query_list.append(sum_query)
            
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Chat",
            "state": board_state,
            "summary": summary_text,
            "question": question_text
        })
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        retrieved_mems, observation_prompt = self._run_in_game_memory_retrieval(observations, summary_text, question_text, env_name, observation_prompt)

        if env_name == 'cooperative_negotiation':
            from gamingbench.prompts.chat_prompts import COOP_CHAT_INSTRUCTION as instruction
        else:
            from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION as instruction
            
        observation_prompt = observation_prompt + '\n\n' + instruction
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        max_retries = 3
        message = ""
        for attempt in range(max_retries):
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            query_list.append(query)
            
            if attempt == 0:
                self.logger.info(f'Chat Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Chat Raw Response (Attempt {attempt+1}): {responses}')
            
            message = strip_chat_tags(strip_thinking_block(responses[0]).strip())
            
            if message:
                self.logger.info(f"Chat Generated: {message}")
                break
            else:
                error_msg = "Failed to extract a valid chat message. You must output a non-empty message wrapped by <chat>...</chat>."
                if attempt < max_retries - 1:
                    msgs.append({"role": "assistant", "content": responses[0]})
                    msgs.append({"role": "user", "content": error_msg})
                    
        self.current_trajectory[-1]["action"] = f"[Chat] {message}"
        
        # Answer Assessment for chat
        answered, memory_conclusion, assess_query = self._assess_answer(retrieved_mems, question_text)
        if assess_query:
            query_list.append(assess_query)
            
        mem_text = "\n".join([f"- {m['content']}" for m in retrieved_mems]) if retrieved_mems else ""
        self.question_log.append({
            "round": observations.get('game_round', self.move_count),
            "question": question_text,
            "retrieved_memory_ids": [m["id"] for m in retrieved_mems],
            "retrieved_memories_text": mem_text if retrieved_mems else "No relevant memories found.",
            "answered": answered,
            "memory_conclusion": memory_conclusion
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
        
        # If batch_mode is True, we do NOT process the memory or write to disk right now.
        if self.batch_mode:
            # Instead, we package the game's data and return it to the main runner script.
            # The main script will collect these from all parallel games and feed them to `flush_batch_updates()` later.
            self._last_batch_result = {
                "opponent_key": self.current_opponent_key,
                "question_log": question_log_data,
                "game_history": game_history
            }
            # Exit early so we don't accidentally update the memory bank concurrently.
            return

        # If batch_mode is False, we process the game immediately (Sequential mode).
        # First, run the Question Review LLM prompt to see if we can extract any new factual evidence from the game log.
        evidence_items = self._process_question_log(self.current_opponent_key, question_log_data, game_history)
        
        # If the Question Review LLM successfully extracted new evidence...
        if evidence_items:
            # ...pass that evidence into the Route & Modify LLM prompt to actually synthesize it into the memory bank.
            self._process_evidence_items(self.current_opponent_key, evidence_items)
            
        # Finally, if we have a valid path to save to (meaning we aren't a temporary clone with /dev/null)...
        if self.store_path != '/dev/null':
            # Write the updated memory bank back to disk so it's ready for the next game.
            self.store.save(self.store_path)
            
    def _process_evidence_items(self, opponent_key, evidence_items):
        """Unified logic to route and modify evidence into the memory store."""
        # 1. Early exit if there is no evidence to process.
        if not evidence_items:
            return
            
        new_evidence_blocks = []
        all_relevant_mems = {}
        
        # 2. Iterate through all the newly extracted evidence pieces
        for item in evidence_items:
            # 3. Save the new evidence directly to the Layer 1 store. 
            # This persists the raw, low-level factual observation and generates a unique evidence ID.
            ev_id = self.store.add_evidence(
                key=opponent_key,
                content=item["content"],
                observation=item["observation"],
                game_id=f"game_{getattr(self, 'game_count', 0)}"
            )
            
            # 4. We want to find which existing high-level memories (Layer 2) are related to this evidence.
            # We embed the low-level evidence content into a vector to use as a semantic query.
            ev_vec = self.embedder.encode(item["content"], is_query=True)
            
            # 5. Retrieve the top 'K' most semantically similar memories from the Layer 2 store.
            relevant_mems = self.store.find_relevant_memories(opponent_key, ev_vec, top_k=self.post_game_top_k)
            
            # 6. Deduplicate the retrieved memories across all evidence items.
            # We store them in a dictionary keyed by memory ID so we only feed unique memories to the LLM.
            for m in relevant_mems:
                if m["id"] not in all_relevant_mems:
                    all_relevant_mems[m["id"]] = m
                    
            # 7. Format the new evidence block to be injected into the LLM prompt.
            new_evidence_blocks.append(f"EVIDENCE ID: {ev_id}\nCONTENT: {item['content']}\nOBSERVATION: {item['observation']}")
            
        # 8. Combine all formatted new evidence blocks into a single string separated by newlines.
        new_evidence_text = "\n\n---\n\n".join(new_evidence_blocks)
        
        # 9. Extract the deduplicated Layer 2 memories as a flat list.
        existing_memories = list(all_relevant_mems.values())
        
        # 10. Format the existing memories so the LLM can see their current IDs, Content, and what evidence supports them.
        if existing_memories:
            formatted_memories = []
            for mem in existing_memories:
                m_id = mem['id']
                content = mem['content']
                ev_ids = mem['evidence_ids']
                
                # Fetch the actual content for each evidence ID
                ev_list = []
                for eid in ev_ids:
                    ev = self.store.get_evidence(opponent_key, eid)
                    if ev:
                        ev_list.append(f"  - [{eid}] {ev['content']}")
                ev_str = "\n".join(ev_list) if ev_list else "  (No evidence)"
                
                formatted_memories.append(f"Memory ID: {m_id}\nContent: {content}\nSupported by Evidence:\n{ev_str}")
            existing_memories_text = "\n\n---\n\n".join(formatted_memories)
        else:
            existing_memories_text = "No existing memories."
            
        # 11. Construct the final LLM prompt, asking it to route the new evidence into the existing memories 
        # (by updating them) or to create entirely new memories.
        prompt = ROUTE_AND_MODIFY_PROMPT.format(
            new_evidence_list=new_evidence_text,
            existing_memories=existing_memories_text
        )
        
        # 12. Build the message array and query the LLM.
        messages = [
            {"role": "system", "content": "You are a strategic memory database manager."},
            {"role": "user", "content": prompt}
        ]
        
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        raw_gen = strip_thinking_block(raw_resp)
        
        # 13. Log the prompt and response for debugging purposes.
        self.logger.info("=== ROUTE & MODIFY ===")
        self.logger.info(f"PROMPT:\n{messages[1]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{raw_gen}")
        self.logger.info("======================")
        
        # 14. Parse the LLM's JSON output to extract its routing decisions.
        parsed_json = extract_json_block(raw_gen)
        decisions = parsed_json.get("routing_decisions", [])
        
        if not isinstance(decisions, list):
            self.logger.error(f"Failed to parse routing_decisions as list: {decisions}")
            return
            
        # 15. Iterate through each decision the LLM made.
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
                
            route = decision.get("routing", "").strip().upper()
            new_mem_content = decision.get("memory_content", "").strip()
            ev_ids_used = decision.get("evidence_ids_used", [])
            
            # 16. Sanitize the evidence IDs returned by the LLM to ensure they are valid.
            if not isinstance(ev_ids_used, list):
                ev_ids_used = []
            valid_ev_ids = [x for x in ev_ids_used if isinstance(x, str) and x.startswith('ev_')]
            
            # 17. Execute the routing decision.
            if route and new_mem_content:
                # If the LLM decided to update an existing memory:
                if route.startswith("ADD_TO_"):
                    mem_id = route.replace("ADD_TO_", "").strip()
                    # Verify the memory actually exists in our store
                    if self.store.get_memory(opponent_key, mem_id):
                        # Generate a new embedding vector for the updated memory content
                        vec = self.embedder.encode(new_mem_content, is_query=False)
                        # Save the updated memory back to the Layer 2 store
                        self.store.update_memory(
                            key=opponent_key,
                            memory_id=mem_id,
                            new_content=new_mem_content,
                            new_evidence_ids=valid_ev_ids,
                            vec=vec,
                            max_evidence_per_memory=self.max_evidence_per_memory
                        )
                # If the LLM decided the evidence warranted a completely new memory:
                elif route == "CREATE_NEW":
                    # Generate an embedding vector for the new memory content
                    vec = self.embedder.encode(new_mem_content, is_query=False)
                    # Create the new memory in the Layer 2 store
                    self.store.add_memory(
                        key=opponent_key,
                        content=new_mem_content,
                        evidence_ids=valid_ev_ids,
                        vec=vec,
                        max_evidence_per_memory=self.max_evidence_per_memory
                    )

    def _process_question_log(self, opponent_key, question_log, game_history):
        """Processes each question asked during the game to verify answers and patch memory gaps."""
        if not question_log:
            return

        formatted_questions = []
        question_map = {}
        for idx, q_entry in enumerate(question_log):
            q_id = f"Q{idx+1}"
            question_map[q_id] = q_entry
            
            formatted_q = f"Question ID: {q_id}\n"
            formatted_q += f"Question: \"{q_entry['question']}\"\n"
            formatted_q += f"ANSWERED IN-GAME: {q_entry['answered']}\n"
            if 'memory_conclusion' in q_entry:
                formatted_q += f"MEMORY CONCLUSION: {q_entry['memory_conclusion']}\n"
            formatted_q += f"Retrieved Memories:\n{q_entry.get('retrieved_memories_text', 'None')}\n"
            formatted_questions.append(formatted_q)
            
        question_log_text = "\n---\n".join(formatted_questions)
        
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
        
        for review in reviews:
            if not isinstance(review, dict):
                continue
                
            q_id = review.get("question_id")
            if q_id not in question_map:
                continue
                
            q_entry = question_map[q_id]
            corr = review.get("correct", True)
            is_undeterminable = False
            
            if isinstance(corr, str):
                if corr.lower() == 'undeterminable':
                    is_undeterminable = True
                    corr = None
                else:
                    corr = corr.lower() == 'true' or corr.lower() == 'yes'
                
            evidence_list = review.get("evidence", [])
            if not isinstance(evidence_list, list) or len(evidence_list) == 0:
                continue
                
            if q_entry["answered"]:
                if is_undeterminable:
                    for ev in evidence_list:
                        if isinstance(ev, dict):
                            ev["observation"] = f"Neutral evidence for unverifiable assumption in question: '{q_entry['question']}'. Original observation: {ev.get('observation', '')}"
                            all_new_evidence.append(ev)
                elif not corr:
                    for ev in evidence_list:
                        if isinstance(ev, dict):
                            ev["observation"] = f"Corrective evidence for false assumption in question: '{q_entry['question']}'. Original observation: {ev.get('observation', '')}"
                            all_new_evidence.append(ev)
            else:
                for ev in evidence_list:
                    if isinstance(ev, dict):
                        ev["observation"] = f"Answer to previously unanswered question: '{q_entry['question']}'. Original observation: {ev.get('observation', '')}"
                        all_new_evidence.append(ev)
                        
        return all_new_evidence

    def flush_batch_updates(self, gradient_data: list) -> None:
        if not gradient_data:
            return
            
        target_key = None
        all_batch_evidence = []
        
        for data in gradient_data:
            if isinstance(data, dict):
                target_key = data.get("opponent_key", target_key)
                q_log = data.get("question_log", [])
                gh = data.get("game_history", "")
                
                if target_key and q_log and gh:
                    ev_items = self._process_question_log(target_key, q_log, gh)
                    if ev_items:
                        all_batch_evidence.extend(ev_items)
                        
        if target_key and all_batch_evidence:
            self._process_evidence_items(target_key, all_batch_evidence)
                    
        if target_key and self.store_path != '/dev/null':
            self.store.save(self.store_path)
