import os
import re
import copy
import json
import threading
import concurrent.futures
from typing import List, Dict, Any

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.two_layer_store import TwoLayerStore
from gamingbench.ltm.two_layer_prompts import (
    OBS_SUMMARIZE_PROMPT,
    EVIDENCE_EXTRACT_PROMPT,
    ROUTE_AND_MODIFY_PROMPT,
    EVIDENCE_INJECTION_PROMPT
)
from gamingbench.prompts.observation_prompts import construct_observation_prompt, construct_game_intro
from gamingbench.prompts.system_prompts import construct_system_prompt
from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags

def extract_json_block(text: str) -> dict:
    """Safely extracts and parses a JSON block from LLM output, handling markdown wrappers."""
    text = text.strip()
    # Try to find a markdown json block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.IGNORECASE | re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # Fallback to finding the first { and last }
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

class EvidenceMemoryAgent(PromptAgent):
    """
    Agent that maintains a 2-layer memory store.
    It summarizes the game state in-game to query memories, and runs a post-game 
    batch process to extract raw trajectory evidence and route it into consolidated memories.
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
        super(EvidenceMemoryAgent, self).__init__(config, **kwargs)
        
        self.batch_mode = getattr(config, "batch_mode", False)
        
        # Pull embedding configurations
        self.embed_model = getattr(config, "embed_model", "Qwen/Qwen3-Embedding-0.6B")
        self.embed_gpu_id = getattr(config, "embed_gpu_id", 0)
        self.embed_max_length = getattr(config, "embed_max_length", 8192)
        self.embed_instruction = getattr(config, "embed_instruction", "Given a summary of the opponent's behavior, find the most applicable strategic memory")
        self.embed_use_flash_attn = getattr(config, "embed_use_flash_attn", True)
        
        # Initialize or retrieve the shared embedder
        self.embedder = self._get_embedder(
            self.embed_model, 
            self.embed_gpu_id, 
            self.embed_max_length, 
            self.embed_instruction, 
            self.embed_use_flash_attn
        )
        
        # Memory hyperparameters
        self.in_game_top_k = getattr(config, "in_game_top_k", 3)
        self.post_game_top_k = getattr(config, "post_game_top_k", 6)
        self.max_evidence_per_memory = getattr(config, "max_evidence_per_memory", 6)
        
        # Initialize the 2-layer store
        self.store_path = getattr(config, "two_layer_store_path", "two_layer_store.json")
        self.store = TwoLayerStore()
        
        # Load existing store if available
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)
            
        # Agent state tracking
        self.current_opponent_key = None
        self.current_game_intro = None
        self.move_count = 0
        self.current_trajectory = []
        self._last_batch_result = None

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
        """Called at the start of a new game to reset turn counters and tracking."""
        # Only reload from disk if not in batch mode (batch mode handles caching explicitly)
        if not self.batch_mode:
            if os.path.exists(self.store_path): 
                self.store.load(self.store_path)
            
        self.game_count = getattr(self, 'game_count', 0) + 1
        
        self.current_trajectory = []
        self.move_count = 0
        self.current_game_intro = game_intro
        
        # Handle hive mode / separate memory modes
        if isinstance(opponent_key, list):
            self.current_opponent_keys = opponent_key
            self.memory_mode = 'separate'
            self.current_opponent_key = opponent_key[0] if len(opponent_key) == 1 else None
        else:
            self.current_opponent_keys = [opponent_key]
            self.memory_mode = 'combined'
            self.current_opponent_key = opponent_key

    def __deepcopy__(self, memo):
        """Custom deepcopy to ensure the embedder singleton isn't erroneously duplicated during batch cloning."""
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

    def _generate_in_game_summary(self, observations):
        """Runs the pre-action LLM call to summarize the board state and strategic intent."""
        sys_prompt, obs_prompt = PromptAgent._build_prompts(self, observations)
        
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': f"--- CURRENT GAME STATE ---\n{obs_prompt}\n\n{OBS_SUMMARIZE_PROMPT}"}
        ]
        
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        stripped_resp = strip_thinking_block(raw_resp)
        
        self.logger.info("=== IN-GAME SUMMARY GENERATION ===")
        self.logger.info(f"PROMPT:\n{messages[1]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{stripped_resp}")
        self.logger.info("==================================")
        
        # Parse out the JSON summary
        parsed_json = extract_json_block(stripped_resp)
        summary = parsed_json.get("summary", stripped_resp.strip())
        
        return summary, query

    def step(self, observations):
        """Main action loop for the agent on its turn."""
        self.move_count += 1
        query_list = []
        
        # 1. Generate the observation summary
        summary_text, sum_query = self._generate_in_game_summary(observations)
        if sum_query:
            query_list.append(sum_query)
            
        # Log the current step for the post-game trajectory
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Action",
            "state": board_state,
            "summary": summary_text
        })
        
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        
        # 2. Memory Retrieval
        if self.current_opponent_key:
            # Embed the generated summary and retrieve memories
            query_vec = self.embedder.encode(summary_text, is_query=True)
            relevant_mems = self.store.find_relevant_memories(self.current_opponent_key, query_vec, top_k=self.in_game_top_k)
            
            if relevant_mems:
                mem_texts = [m['content'] for m in relevant_mems]
                text_blob = "\n\n---\n\n".join(mem_texts)
            else:
                text_blob = "None."
                
            injection = EVIDENCE_INJECTION_PROMPT.format(retrieved_memories=text_blob)
            
            observation_prompt = observation_prompt.replace(board_state, injection + "\n\n" + board_state, 1)

        # 3. Action Generation
        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead."
            
        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        # Standard retry loop for generating valid moves
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
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        """Optional chat step loop, structurally similar to step()."""
        if not getattr(self, 'enable_chat', False):
            return "", None
            
        self.move_count += 1
        query_list = []
        
        # Generate chat-specific summary
        summary_text, sum_query = self._generate_in_game_summary(observations)
        if sum_query:
            query_list.append(sum_query)
            
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)
        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Chat",
            "state": board_state,
            "summary": summary_text
        })
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)
        
        # Memory Retrieval
        if self.current_opponent_key:
            query_vec = self.embedder.encode(summary_text, is_query=True)
            relevant_mems = self.store.find_relevant_memories(self.current_opponent_key, query_vec, top_k=self.in_game_top_k)
            
            if relevant_mems:
                mem_texts = [m['content'] for m in relevant_mems]
                text_blob = "\n\n---\n\n".join(mem_texts)
            else:
                text_blob = "None."
                
            injection = EVIDENCE_INJECTION_PROMPT.format(retrieved_memories=text_blob)
            
            game_intro = construct_game_intro(env_name, enable_chat=getattr(self, 'enable_chat', False), game_config=getattr(self, 'game_config', None))
            observation_prompt = observation_prompt.replace(game_intro, game_intro + "\n\n" + injection, 1)

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
        # We generally return the last query for compat with chat_step signature (returns str, Any)
        return message, query_list[-1] if query_list else None

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """Called by the framework after a game ends to update long-term memory."""
        if not self.current_opponent_key:
            return

        # Personalize the trajectory view for the agent
        player_index = getattr(self, 'current_player_index', None)
        if player_index is not None:
            game_history = game_history.replace(f"Player {player_index}", "You")
            
        self.logger.info('-' * 20 + f'{self.agent_name} Post-Game Evidence Update' + '-' * 20)
        
        # 1. Extract raw Evidence from the full game history
        messages = [
            {"role": "user", "content": EVIDENCE_EXTRACT_PROMPT.format(game_trajectory=game_history)}
        ]
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        raw_evidence_gen = strip_thinking_block(raw_resp)
        
        self.logger.info("=== POST-GAME EVIDENCE EXTRACTION ===")
        self.logger.info(f"PROMPT:\n{messages[0]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{raw_evidence_gen}")
        self.logger.info("=======================================")
        
        evidence_items = []
        # Parse the JSON evidence array
        parsed_json = extract_json_block(raw_evidence_gen)
        ev_array = parsed_json.get("evidence", [])
        
        for item in ev_array:
            if "content" in item and "observation" in item:
                evidence_items.append({"content": item["content"].strip(), "observation": item["observation"].strip()})
            
        self.logger.info(f"Extracted {len(evidence_items)} evidence items.")

        # In batch mode, we defer processing and just stash the extracted evidence
        if self.batch_mode:
            self._last_batch_result = {
                "opponent_key": self.current_opponent_key,
                "evidence_items": evidence_items
            }
            return

        # Otherwise, process it immediately
        self._process_evidence_items(self.current_opponent_key, evidence_items)
        if self.store_path != '/dev/null':
            self.store.save(self.store_path)

    def _format_memory_context(self, opponent_key, memories):
        """Formats the retrieved memories and their attached evidence for the LLM prompt."""
        if not memories:
            return "No existing memories."
        parts = []
        for m in memories:
            ev_list = []
            for eid in m["evidence_ids"]:
                ev = self.store.get_evidence(opponent_key, eid)
                if ev:
                    ev_list.append(f"  - [{eid}] {ev['content']}")
            ev_str = "\n".join(ev_list) if ev_list else "  (No evidence)"
            parts.append(f"MEMORY ID: {m['id']}\nCONTENT: {m['content']}\nSUPPORTING EVIDENCE:\n{ev_str}")
        return "\n\n".join(parts)

    def _process_evidence_items(self, opponent_key, evidence_items):
        """The core memory routing pipeline: embeds evidence, retrieves memories, updates store."""
        if not evidence_items:
            return

        new_evidence_blocks = []
        all_relevant_mems = {}
        
        for item in evidence_items:
            # 1. Add to Layer 1 (returns the new evidence_id)
            ev_id = self.store.add_evidence(
                key=opponent_key,
                content=item["content"],
                observation=item["observation"],
                game_id=f"game_{getattr(self, 'game_count', 0)}"
            )
            
            # 2. Retrieve Layer 2 memories by embedding the new evidence content
            ev_vec = self.embedder.encode(item["content"], is_query=True)
            relevant_mems = self.store.find_relevant_memories(opponent_key, ev_vec, top_k=self.post_game_top_k)
            
            for m in relevant_mems:
                if m["id"] not in all_relevant_mems:
                    all_relevant_mems[m["id"]] = m
                    
            new_evidence_blocks.append(f"EVIDENCE ID: {ev_id}\nCONTENT: {item['content']}\nOBSERVATION: {item['observation']}")
            
        # 3. Route & Modify LLM Call (batched prompt)
        new_evidence_list = "\n\n".join(new_evidence_blocks)
        mem_context = self._format_memory_context(opponent_key, list(all_relevant_mems.values()))
        
        prompt = ROUTE_AND_MODIFY_PROMPT.format(
            new_evidence_list=new_evidence_list,
            existing_memories=mem_context
        )
        
        env_name = getattr(self, 'current_game_name', None)
        messages = []
        if env_name:
            try:
                messages.append({"role": "system", "content": construct_system_prompt(env_name)})
            except KeyError:
                pass
        messages.append({"role": "user", "content": prompt})
        
        responses, query = self.llm_query(messages, n=1, stop=None, prompt_type='move')
        raw_resp = responses[0]
        raw_gen = strip_thinking_block(raw_resp)
        
        self.logger.info("=== ROUTE & MODIFY ===")
        self.logger.info(f"PROMPT:\n{messages[1]['content']}")
        self.logger.info(f"RAW ANSWER:\n{raw_resp}")
        self.logger.info(f"STRIPPED ANSWER:\n{raw_gen}")
        self.logger.info("======================")
        
        log_dir = os.path.dirname(self.store_path)
        if log_dir and log_dir != '/dev/null':
            ev_log_file = os.path.join(log_dir, f"{self.agent_name}_evidence_processing.log")
            try:
                with open(ev_log_file, "a") as f:
                    f.write("=== ROUTE & MODIFY ===\n")
                    f.write(f"PROMPT:\n{messages[1]['content']}\n")
                    f.write(f"RAW ANSWER:\n{raw_resp}\n")
                    f.write(f"STRIPPED ANSWER:\n{raw_gen}\n")
                    f.write("=======================\n\n")
            except Exception as e:
                self.logger.error(f"Failed to write to evidence processing log: {e}")
        
        # 4. Parse Decisions out of the unified JSON array
        parsed_json = extract_json_block(raw_gen)
        decisions = parsed_json.get("routing_decisions", [])
        
        if not isinstance(decisions, list):
            self.logger.error(f"Failed to parse routing_decisions as list: {decisions}")
            return
            
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
                
            route = decision.get("routing", "").strip().upper()
            new_mem_content = decision.get("memory_content", "").strip()
            ev_ids_used = decision.get("evidence_ids_used", [])
            
            if not isinstance(ev_ids_used, list):
                ev_ids_used = []
            valid_ev_ids = [x for x in ev_ids_used if isinstance(x, str) and x.startswith('ev_')]
            
            if route and new_mem_content:
                # Branch A: Update an existing memory
                if route.startswith("ADD_TO_"):
                    mem_id = route.replace("ADD_TO_", "").strip().lower()
                    if self.store.get_memory(opponent_key, mem_id):
                        vec = self.embedder.encode(new_mem_content, is_query=False) # Embed the updated content
                        self.store.update_memory(
                            key=opponent_key,
                            memory_id=mem_id,
                            new_content=new_mem_content,
                            new_evidence_ids=valid_ev_ids,
                            vec=vec,
                            max_evidence_per_memory=self.max_evidence_per_memory
                        )
                # Branch B: Create a brand new memory
                elif route == "CREATE_NEW":
                    vec = self.embedder.encode(new_mem_content, is_query=False) # Embed the new content
                    self.store.add_memory(
                        key=opponent_key,
                        content=new_mem_content,
                        evidence_ids=valid_ev_ids,
                        vec=vec,
                        max_evidence_per_memory=self.max_evidence_per_memory
                    )

    def flush_batch_updates(self, gradient_data: list) -> None:
        """Called centrally in batch mode to process all queued evidence items at once."""
        if not gradient_data:
            return
            
        all_evidence = []
        target_key = None
        for data in gradient_data:
            if isinstance(data, dict):
                target_key = data.get("opponent_key", target_key)
                all_evidence.extend(data.get("evidence_items", []))
                
        if target_key and all_evidence:
            self._process_evidence_items(target_key, all_evidence)
            if self.store_path != '/dev/null':
                self.store.save(self.store_path)
