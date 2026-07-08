import sys
sys.modules['transformer_engine'] = None

import json
import os
import re
from typing import List, Tuple, Dict, Any

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.reasoning_bank_prompts import SUCCESSFUL_MEMORY_SI, FAILED_MEMORY_SI, MEMORY_INJECTION_PROMPT

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
except ImportError:
    pass

class ReasoningBankAgent(PromptAgent):
    def __init__(self, config, **kwargs):
        super(ReasoningBankAgent, self).__init__(config, **kwargs)
        self.memory_bank_path = getattr(config, "memory_bank_path", "reasoning_bank.json")
        self.num_memories = getattr(config, "num_memories", 3)
        self.max_bank_size = getattr(config, "max_bank_size", 50)
        
        self.batch_mode = False
        self._last_batch_result = None
        self.current_trajectory = []
        
        self.memory_bank: List[Dict] = []
        self.current_memories: List[Dict] = []
        self._faiss_index = None

    def set_storage_dir(self, storage_dir):
        """Called by openspiel_adapter to align storage with the run's experiment folder."""
        self.memory_bank_path = os.path.join(storage_dir, os.path.basename(self.memory_bank_path))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Prevent deepcopying the PyTorch embedding model inside the FAISS index, which causes CUDA OOM
        if '_faiss_index' in state:
            state['_faiss_index'] = None
        return state

    def reset_game_state(self, opponent_key, game_intro):
        """Called at the start of a game to prepare tracking and retrieve memories."""
        if not self.batch_mode:
            self._load_memory_bank()
            self._rebuild_faiss_index()

        self.current_trajectory = []
        
        # Retrieve memories
        self.current_memories = self._retrieve_memories(game_intro)

    def _load_memory_bank(self):
        if os.path.exists(self.memory_bank_path):
            with open(self.memory_bank_path, 'r', encoding='utf-8') as f:
                try:
                    self.memory_bank = json.load(f)
                except json.JSONDecodeError:
                    self.memory_bank = []
        else:
            self.memory_bank = []

    def _save_memory_bank(self):
        with open(self.memory_bank_path, 'w', encoding='utf-8') as f:
            json.dump(self.memory_bank, f, indent=4)

    def _rebuild_faiss_index(self):
        if not self.memory_bank:
            self._faiss_index = None
            return

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        docs = []
        for i, memory in enumerate(self.memory_bank):
            # Combine fields to give FAISS good semantic context
            content_str = f"Title: {memory.get('title', '')}\nDescription: {memory.get('description', '')}\nContent: {memory.get('content', '')}"
            docs.append(Document(page_content=content_str, metadata={"index": i}))
        
        self._faiss_index = FAISS.from_documents(docs, embeddings)

    def _retrieve_memories(self, query: str) -> List[Dict]:
        if not self.memory_bank:
            return []
            
        if getattr(self, '_faiss_index', None) is None:
            self._rebuild_faiss_index()
            
        if not self._faiss_index:
            return []
            
        k = min(self.num_memories, len(self.memory_bank))
        docs = self._faiss_index.similarity_search(query, k=k)
        retrieved = []
        for doc in docs:
            idx = doc.metadata["index"]
            retrieved.append(self.memory_bank[idx])
        return retrieved

    def _format_memories_block(self) -> str:
        if not self.current_memories:
            return ""
        
        items_str = ""
        for i, memory in enumerate(self.current_memories):
            items_str += f"# Memory {i+1}\n"
            items_str += f"## Title: {memory.get('title', '')}\n"
            items_str += f"## Description: {memory.get('description', '')}\n"
            items_str += f"## Content: {memory.get('content', '')}\n\n"
            
        return MEMORY_INJECTION_PROMPT.format(memory_text=items_str.strip())

    def _build_prompts(self, observations):
        system_prompt, observation_prompt = super()._build_prompts(observations)
        
        memories_block = self._format_memories_block()
        if memories_block:
            # Inject memory block after the game rules/intro
            # Usually the game intro is at the top of the observation_prompt
            if "Past rounds:" in observation_prompt:
                parts = observation_prompt.split("Past rounds:")
                new_observation_prompt = parts[0] + "\n" + memories_block + "\n\nPast rounds:" + parts[1]
                observation_prompt = new_observation_prompt
            else:
                # Fallback: prepend to the observation prompt
                observation_prompt = memories_block + "\n\n" + observation_prompt
                
        return system_prompt, observation_prompt

    def step(self, observations):
        self.logger.info('-' * 20 + f'{self.agent_name} Begin' + '-' * 20)
        query_list = []

        system_prompt, observation_prompt = self._build_prompts(observations)

        if getattr(self, "think_further", False):
            observation_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead. Anticipate the opponent's likely responses to your move, and consider long-term strategic implications rather than just immediate tactical gains."

        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        # Save regex for move extraction
        from gamingbench.prompts.regex_and_format import get_step_env_regex_and_format
        regex, _ = get_step_env_regex_and_format(observations.get('env_name', ''))

        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""
        final_responses = None

        for attempt in range(max_retries):
            responses, query = self.llm_query(
                msgs, n=self.num_generations, stop=None, prompt_type='move', enable_thinking=True)
            query_list.append(query)

            if attempt == 0:
                self.logger.info(f'Prompt: {observation_prompt}')
            self.logger.info(f'Response (Attempt {attempt+1}): {responses}')
            final_responses = responses

            moves = self.parse_with_regex(responses, regex)
            if len(moves) != 0:
                move = self.post_processing(moves, majority_vote=getattr(self, "voting", False))
                # Normalize brackets and asterisks from move and valid_moves list to handle potential mismatch
                def clean_action(act):
                    return act.replace('<', '').replace('>', '').replace('*', '').strip()
                cleaned_move = clean_action(move)
                matched_valid_move = None
                for m in valid_moves:
                    if clean_action(m) == cleaned_move:
                        matched_valid_move = m
                        break
                if not valid_moves or matched_valid_move is not None:
                    if matched_valid_move is not None:
                        move = matched_valid_move
                    break
                else:
                    error_msg = f"Invalid move '{move}'. Your move must be one of the legal actions: {valid_moves}. Please try again."
            else:
                move = ""
                error_msg = f"Failed to extract a valid move format. You must output your action wrapped by <>, i.e., <[a-c][1-8]->[a-c][1-8]>. Legal actions: {valid_moves}. Please try again."
            
            if attempt < max_retries - 1:
                self.logger.warning(error_msg)
                msgs.append({"role": "assistant", "content": responses[0]})
                msgs.append({"role": "user", "content": error_msg})

        if final_responses:
            self.current_trajectory.append(f"Prompt:\n{observation_prompt}\n\nResponse:\n{final_responses[0]}")

        self.logger.info('-' * 20 + f'{self.agent_name} End' + '-' * 20)
        return move, query_list

    def chat_step(self, observations, chat_history_str: str):
        if not getattr(self, "enable_chat", False):
            return "", None
            
        from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION
        
        self.logger.info('-' * 20 + f'{self.agent_name} Chat Generation' + '-' * 20)
        
        observations['chat_context'] = chat_history_str
        system_prompt, observation_prompt = self._build_prompts(observations)
        observation_prompt = observation_prompt + '\n\n' + CHAT_INSTRUCTION
        msgs = self.construct_init_messages(system_prompt, observation_prompt)
        
        try:
            from gamingbench.utils.utils import strip_thinking_block, strip_chat_tags
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move', enable_thinking=True)
            message = strip_thinking_block(responses[0]).strip()
            message = strip_chat_tags(message)
            self.logger.info(f"Chat Generated: {message}")
            return message, query
        except Exception as e:
            self.logger.error(f"Chat generation failed: {e}")
            return "", None

    def post_game_update(self, game_history: str, final_board_state: str, env_name: str):
        # Determine if we won
        # Look for "Your score=X, Opponent score=Y"
        score_pattern = r"Your score=([-\d.]+), Opponent score=([-\d.]+)"
        matches = list(re.finditer(score_pattern, game_history))
        won = False
        if matches:
            last_match = matches[-1]
            my_score = float(last_match.group(1))
            opp_score = float(last_match.group(2))
            won = (my_score > opp_score)
            
        result = {
            "trajectory": self.current_trajectory.copy(),
            "won": won,
            "env_name": env_name,
            "game_intro": game_history.split("Past rounds:")[0].strip() if "Past rounds:" in game_history else ""
        }

        if self.batch_mode:
            self._last_batch_result = result
            return
            
        # Not in batch mode, do extraction immediately
        new_items = self._extract_memories_from_trajectory(result)
        self._update_memory_bank(new_items)
        self._rebuild_faiss_index()

    def _extract_memories_from_trajectory(self, result: dict) -> List[Dict]:
        trajectory_list = result["trajectory"]
        if not trajectory_list:
            return []
            
        trajectory_text = "\n\n".join(trajectory_list)
        won = result["won"]
        env_name = result["env_name"]
        game_intro = result["game_intro"]
        
        system_msg = SUCCESSFUL_MEMORY_SI if won else FAILED_MEMORY_SI
        user_msg = f"Game: {env_name}\n\nGame Introduction:\n{game_intro}\n\nFull Game Trajectory:\n{trajectory_text}"
        
        msgs = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            self.logger.info(f"Extracting ReasoningBank memories (won={won})")
            responses, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move', enable_thinking=True)
            output = responses[0]
            
            from gamingbench.utils.utils import strip_thinking_block
            output = strip_thinking_block(output)
            
            # Parse markdown format:
            # # Memory Item i
            # ## Title <title>
            # ## Description <desc>
            # ## Content <content>
            
            extracted = []
            
            # Split by memory item
            items = re.split(r'#\s*Memory\s*Item\s*\d+', output)
            for item in items:
                if not item.strip():
                    continue
                
                title_match = re.search(r'##\s*Title\s*[:\*]*\s*(.+)', item, re.IGNORECASE)
                desc_match = re.search(r'##\s*Description\s*[:\*]*\s*(.+)', item, re.IGNORECASE)
                content_match = re.search(r'##\s*Content\s*[:\*]*\s*(.+)', item, re.IGNORECASE | re.DOTALL)
                
                if title_match and desc_match and content_match:
                    extracted.append({
                        "title": title_match.group(1).strip(),
                        "description": desc_match.group(1).strip(),
                        "content": content_match.group(1).strip(),
                        "game_type": env_name,
                        "status": "success" if won else "fail"
                    })
            
            return extracted
        except Exception as e:
            self.logger.error(f"Memory extraction failed: {e}")
            return []

    def _update_memory_bank(self, new_items: List[Dict]):
        if not new_items:
            return
            
        self._load_memory_bank()
        self.memory_bank.extend(new_items)
        
        if len(self.memory_bank) > self.max_bank_size:
            # FIFO: trim the oldest
            self.memory_bank = self.memory_bank[-self.max_bank_size:]
            
        self._save_memory_bank()

    def flush_batch_updates(self, results: List[dict]):
        """Called by the batch runner when all N games in a batch complete."""
        all_new_items = []
        for res in results:
            if not res:
                continue
            new_items = self._extract_memories_from_trajectory(res)
            all_new_items.extend(new_items)
            
        if all_new_items:
            self._update_memory_bank(all_new_items)
            self._rebuild_faiss_index()
