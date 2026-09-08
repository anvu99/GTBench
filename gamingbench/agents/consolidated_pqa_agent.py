"""
gamingbench/agents/consolidated_pqa_agent.py

ConsolidatedPQA v2 Agent — observation graph-based memory agent that:
  - Injects a pre-built consolidated memory block at every step/chat_step
  - Generates <sus> (suspicious behavior) tags inline with each action/chat
  - After each game: runs ONE memory confirmation LLM call
    (raw <sus> items go directly into the batch accumulator — no per-game LLM)
  - After the batch, runs a 3-step pipeline:
      A. Dedup + edge creation (one LLM call)
      B. Memory synthesis per changed component (one LLM call each)
      C. Final consolidation into injected profile (one LLM call)
"""

import os
import re
import json
import threading
import concurrent.futures
import copy
import time
import numpy as np

from gamingbench.agents.prompt_agent import PromptAgent
from gamingbench.ltm.cpqa_prompts import (
    CPQA_MEMORY_INJECTION, CPQA_MEMORY_INJECTION_EMPTY, CPQA_SUS_SUFFIX,
    CPQA_POST_GAME_CONFIRM_PROMPT, CPQA_POST_GAME_SUS_EXTRACTION_PROMPT,
    CPQA_BATCH_DEDUP_EDGE_PROMPT, CPQA_MEMORY_SYNTHESIS_PROMPT,
    STEP_A_PROMPTS, STEP_B_PROMPTS,
)
from gamingbench.ltm.cpqa_obs_graph import ObsGraphStore, OBS_TYPES, _obs_anchor_text, _format_memory_for_prompt
from gamingbench.prompts.observation_prompts import construct_observation_prompt
from gamingbench.utils.utils import strip_thinking_block


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def extract_json_block(text: str) -> dict:
    """Safely extracts and parses a JSON block from LLM output."""
    text = text.strip()
    match = re.search(r'```json\s*(.*?)\s*```', text, re.IGNORECASE | re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
        else:
            json_str = "{}"
    try:
        parsed = json.loads(json_str)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# SUS tag parser
# ---------------------------------------------------------------------------

def extract_sus_items(text: str) -> list:
    """Parses <sus>...</sus> tag from LLM output. Returns list of strings."""
    matches = re.findall(r'<sus>(.*?)</sus>', text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return []
    raw = matches[-1]
    items = []
    for line in raw.splitlines():
        line = line.strip().lstrip('-').strip()
        if line:
            items.append(line)
    return items


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ConsolidatedPQAAgent(PromptAgent):
    """
    ConsolidatedPQA v2: observation graph memory via inline <sus> generation
    and batch-end consolidation pipeline (Steps A→B→C).
    """

    _log_lock = threading.Lock()

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.batch_mode = getattr(config, "batch_mode", False)
        self.max_memories = getattr(config, "max_memories", 15)
        # Embedding + FAISS settings
        self.embedder_model = getattr(config, "embedder_model", "Qwen/Qwen3-Embedding-0.6B")
        self.embedder_gpu_id = getattr(config, "embedder_gpu_id", 0)
        self.faiss_dedup_threshold = getattr(config, "faiss_dedup_threshold", 0.85)
        self.faiss_edge_threshold = getattr(config, "faiss_edge_threshold", 0.60)
        self.faiss_top_k = getattr(config, "faiss_top_k", 5)  # neighbors shown in Step A

        self.store_path = getattr(config, "cpqa_store_path", "cpqa_obs_graph.json")
        self.store = ObsGraphStore()
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)

        # Lazy-initialized embedder (avoid loading 0.6B at import time)
        self._embedder = None

        # Per-game state
        self.current_opponent_key = None
        self.current_game_intro = None
        self._sus_accumulator: list = []
        self._last_batch_result = None
        self._storage_dir = None

    # ------------------------------------------------------------------
    # Embedder (lazy init)
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-initializes QwenEmbedder on first use."""
        if self._embedder is None:
            try:
                from gamingbench.ltm.qwen_embedder import QwenEmbedder
                self._embedder = QwenEmbedder(
                    model_name=self.embedder_model,
                    gpu_id=self.embedder_gpu_id,
                    instruction="Given a behavioral observation about a game opponent, find semantically similar observations",
                    use_flash_attn=True,
                )
            except Exception as e:
                print(f"Warning: Could not load QwenEmbedder: {e}. Embedding-based dedup disabled.")
                self._embedder = None
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        """Embeds a string. Returns None if embedder unavailable."""
        embedder = self._get_embedder()
        if embedder is None or not text.strip():
            return None
        try:
            return embedder.encode(text, is_query=False)
        except Exception as e:
            print(f"Warning: Embedding failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Framework hooks
    # ------------------------------------------------------------------

    def set_storage_dir(self, storage_dir: str):
        self._storage_dir = storage_dir
        base = os.path.basename(self.store_path)
        self.store_path = os.path.join(storage_dir, base)
        if os.path.exists(self.store_path):
            self.store.load(self.store_path)

    def reset_game_state(self, opponent_key, game_intro):
        if not self.batch_mode:
            if os.path.exists(self.store_path):
                self.store.load(self.store_path)

        if isinstance(opponent_key, list):
            self.current_opponent_key = "+".join(sorted(opponent_key))
        else:
            self.current_opponent_key = opponent_key

        self.current_game_intro = game_intro
        self.current_trajectory = []
        self.move_count = 0
        self._sus_accumulator = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ('store', '_embedder'):
                # Share store and embedder across clones (both are thread-safe)
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_prompt(self, phase_title: str, prompt: str, raw_answer: str):
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

        with self._log_lock:
            try:
                log_dir = None
                if self._storage_dir and self._storage_dir != '/dev/null':
                    log_dir = self._storage_dir
                elif self.store_path and os.path.dirname(self.store_path) not in ('', '/dev/null'):
                    candidate = os.path.dirname(self.store_path)
                    if candidate != '/dev/null':
                        log_dir = candidate

                if log_dir:
                    log_file = os.path.join(log_dir, f"{self.agent_name}_cpqa_processing.log")
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(log_str)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Failed to write CPQA processing log: {e}")

    # ------------------------------------------------------------------
    # In-game memory block injection
    # ------------------------------------------------------------------

    def _get_memory_injection(self) -> str:
        if not self.current_opponent_key:
            return CPQA_MEMORY_INJECTION_EMPTY
        block = self.store.get_consolidated_block(self.current_opponent_key)
        if not block:
            return CPQA_MEMORY_INJECTION_EMPTY
        return CPQA_MEMORY_INJECTION.format(consolidated_block=block)

    # ------------------------------------------------------------------
    # step() — action turn
    # ------------------------------------------------------------------

    def step(self, observations):
        self.move_count += 1
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)

        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)

        memory_injection = self._get_memory_injection()
        observation_prompt = observation_prompt.replace(
            board_state,
            memory_injection + "\n\n" + board_state,
            1
        )

        step_instruct = self.step_prompt_constructor(observations)
        step_prompt = step_instruct['prompt']
        if getattr(self, "think_further", False):
            step_prompt += "\n\nBefore generating your action, carefully think multiple steps ahead."

        # Append SUS suffix
        step_prompt += CPQA_SUS_SUFFIX

        observation_prompt = observation_prompt + '\n' + step_prompt
        regex = step_instruct['regex']
        msgs = self.construct_init_messages(system_prompt, observation_prompt)

        valid_moves = observations.get('legal_moves', [])
        max_retries = 3
        move = ""
        query_list = []

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
            moves = self.parse_with_regex(responses, regex)
            if moves:
                move = self.post_processing(moves, majority_vote=getattr(self, "voting", False))

                def clean_action(act):
                    return act.replace('<', '').replace('>', '').replace('*', '').strip()

                cleaned = clean_action(move)
                matched = next((m for m in valid_moves if clean_action(m) == cleaned), None)
                if not valid_moves or matched is not None:
                    if matched is not None:
                        move = matched
                else:
                    error_parts.append(
                        f"Invalid move '{move}'. Your move must be one of the legal actions: {valid_moves}."
                    )
            else:
                move = ""
                error_parts.append(f"Failed to extract a valid move format. Legal actions: {valid_moves}.")

            if not error_parts:
                sus_items = extract_sus_items(stripped_response)
                self._sus_accumulator.extend(sus_items)
                break

            if attempt < max_retries - 1:
                msgs.append({"role": "assistant", "content": raw_response})
                msgs.append({"role": "user", "content": " ".join(error_parts) + " Please try again."})

        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Action",
            "state": board_state,
            "action": f"[Move] {move}",
        })

        return move, query_list

    # ------------------------------------------------------------------
    # chat_step() — chat turn
    # ------------------------------------------------------------------

    def chat_step(self, observations, chat_history_str: str):
        if not getattr(self, 'enable_chat', False):
            return "", None

        self.move_count += 1
        observations['chat_context'] = chat_history_str
        env_name = observations['env_name']
        board_state = construct_observation_prompt(observations, env_name)

        system_prompt, observation_prompt = PromptAgent._build_prompts(self, observations)

        memory_injection = self._get_memory_injection()
        observation_prompt = observation_prompt.replace(
            board_state,
            memory_injection + "\n\n" + board_state,
            1
        )

        if env_name == 'cooperative_negotiation':
            from gamingbench.prompts.chat_prompts import COOP_CHAT_INSTRUCTION as instruction
        else:
            from gamingbench.prompts.chat_prompts import CHAT_INSTRUCTION as instruction

        observation_prompt = observation_prompt + '\n\n' + instruction
        observation_prompt += CPQA_SUS_SUFFIX

        msgs = self.construct_init_messages(system_prompt, observation_prompt)

        max_retries = 3
        message = ""
        query = None

        for attempt in range(max_retries):
            responses, query = self.llm_query(msgs, n=1, stop=None, prompt_type='move')

            if attempt == 0:
                self.logger.info(f'Chat Prompt: {msgs[1]["content"]}')
            self.logger.info(f'Chat Response (Attempt {attempt+1}): {responses}')

            raw_response = responses[0]
            stripped_response = strip_thinking_block(raw_response)
            self.logger.info(f'Chat Stripped Response (Attempt {attempt+1}):\n{stripped_response}')

            from gamingbench.utils.utils import strip_chat_tags
            message = strip_chat_tags(stripped_response).strip()

            if not message:
                if attempt < max_retries - 1:
                    msgs.append({"role": "assistant", "content": raw_response})
                    msgs.append({
                        "role": "user",
                        "content": "Failed to extract a valid chat message. You must output a non-empty message wrapped by <chat>...</chat>. Please try again."
                    })
                continue

            sus_items = extract_sus_items(stripped_response)
            self._sus_accumulator.extend(sus_items)
            break

        self.current_trajectory.append({
            "round": observations.get('game_round', self.move_count),
            "phase": "Chat",
            "state": board_state,
            "action": f"[Chat] {message}",
        })

        return message, query

    # ------------------------------------------------------------------
    # post_game_update() — confirmation call only, no Step 4 LLM call
    # ------------------------------------------------------------------

    def post_game_update(self, game_history: str, final_board_state: str = "", env_name: str = 'unknown'):
        """
        Per-game update. Removed: Step 4 DAI cleanup LLM call.
        Now only:
          1. Runs memory confirmation call (which bullets occurred this game?)
          2. Stashes raw <sus> items + confirmation result into batch accumulator.
        """
        if not self.current_opponent_key:
            return

        if self.agent_name:
            game_history = game_history.replace(self.agent_name, f"{self.agent_name} (You)")

        opp_key = self.current_opponent_key
        game_rules = self.current_game_intro or ""

        # ---- Memory Confirmation call ----
        confirmation_result = {}
        memories = self.store.get_all_memories(opp_key)
        memory_list_str = "(No memories yet.)"
        if memories:
            memory_list_str = self.store.format_memories_for_confirm(opp_key)
            confirm_prompt = CPQA_POST_GAME_CONFIRM_PROMPT.format(
                game_rules=game_rules,
                game_trajectory=game_history,
                memory_list=memory_list_str,
            )
            msgs = [{"role": "user", "content": confirm_prompt}]
            resp, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            raw = resp[0]
            self._log_prompt("POST-GAME: MEMORY CONFIRMATION", confirm_prompt, raw)
            parsed = extract_json_block(strip_thinking_block(raw))
            confirmation_result = parsed.get("confirmations", [])
        else:
            confirmation_result = []

        # ---- Apply confirmed bullet counts immediately (programmatic) ----
        self._apply_confirmations(opp_key, confirmation_result)

        # ---- Increment games observed ----
        self.store.increment_games_observed(opp_key)

        # ---- Post-game SUS Extraction call ----
        structured_sus = []
        if self._sus_accumulator:
            sus_list_str = "\n".join(f"- {s}" for s in self._sus_accumulator)
            sus_ext_prompt = CPQA_POST_GAME_SUS_EXTRACTION_PROMPT.format(
                game_rules=game_rules,
                game_trajectory=game_history,
                memory_list=memory_list_str,
                raw_sus_items=sus_list_str,
            )
            msgs = [{"role": "user", "content": sus_ext_prompt}]
            resp, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            raw = resp[0]
            self._log_prompt("POST-GAME: SUS EXTRACTION", sus_ext_prompt, raw)
            parsed = extract_json_block(strip_thinking_block(raw))
            structured_sus = parsed.get("structured_observations", [])

        # ---- Programmatically structure new variants from confirmation ----
        structured_new_variants = []
        for c in confirmation_result:
            if not isinstance(c, dict):
                continue
            new_behavior = c.get("new_behavior")
            mem_id = c.get("mem_id")
            if new_behavior and mem_id:
                mem = self.store.get_memory(opp_key, mem_id)
                if mem:
                    m_type = mem.get("type", "UNCONDITIONAL")
                    anchor = mem.get("anchor", "")
                    obs = {"type": m_type}
                    if m_type == "UNCONDITIONAL":
                        obs["behavior"] = new_behavior
                    elif m_type == "CONDITIONAL":
                        obs["game_state"] = anchor
                        obs["action"] = new_behavior
                    elif m_type == "INFERENCE":
                        obs["public_action"] = anchor
                        obs["underlying_driver"] = new_behavior
                    
                    structured_new_variants.append(obs)

        # ---- Stash for batch pipeline ----
        batch_item = {
            "opponent_key": opp_key,
            "structured_obs": structured_sus + structured_new_variants,
        }

        if self.batch_mode:
            self._last_batch_result = batch_item
            return

        # Non-batch: run full pipeline immediately
        self._run_full_pipeline(
            opponent_key=opp_key,
            all_game_data=[batch_item],
            game_rules=game_rules,
        )
        if self.store_path and self.store_path != '/dev/null':
            self.store.save(self.store_path)

    # ------------------------------------------------------------------
    # flush_batch_updates() — Steps A→B→C
    # ------------------------------------------------------------------

    def flush_batch_updates(self, gradient_data: list) -> None:
        """
        Called by main.py after all games in a batch complete.
        gradient_data: list of batch_item dicts, one per game.
        """
        if not gradient_data:
            return

        grouped: dict = {}
        for item in gradient_data:
            if not isinstance(item, dict):
                continue
            key = item.get("opponent_key")
            if key:
                grouped.setdefault(key, []).append(item)

        for opp_key, game_data_list in grouped.items():
            game_rules = self.current_game_intro or ""
            self._run_full_pipeline(opp_key, game_data_list, game_rules)

        if self.store_path and self.store_path != '/dev/null':
            self.store.save(self.store_path)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def _run_full_pipeline(self, opponent_key: str, all_game_data: list, game_rules: str):
        """Runs Steps A→B→C for one opponent."""
        self._run_step_A(opponent_key, all_game_data, game_rules)
        self._run_step_C(opponent_key, game_rules)

    # ------------------------------------------------------------------
    # Step A: Dedup + Edge Creation
    # ------------------------------------------------------------------

    def _run_step_A(self, opponent_key: str, all_game_data: list, game_rules: str):
        """
        Collects all structured observations from all games,
        embeds them, runs FAISS dedup/candidate retrieval, then one LLM
        call to deduplicate across the batch and assign to components.
        """
        # 1. Collect all structured observation items
        structured_items = []
        for game_data in all_game_data:
            for obs in game_data.get("structured_obs", []):
                structured_items.append(obs)

        if not structured_items:
            # Nothing new to process (no new notes or variants).
            # Any incremented counts from post-game confirmations will be automatically 
            # picked up by Step C. We can just return.
            return

        # 2. Format items with temp IDs and group them by type
        grouped_items = {t: [] for t in OBS_TYPES}
        for i, item in enumerate(structured_items):
            temp_id = f"new_{i+1:03d}"
            item["temp_id"] = temp_id
            obs_type = item.get("type")
            if obs_type in grouped_items:
                grouped_items[obs_type].append(item)

        # Helper function for a single observation type
        def _process_obs_type(obs_type: str, items: list) -> dict:
            obs_lines = []
            for item in items:
                item_copy = dict(item)
                item_copy["id"] = item_copy.pop("temp_id")
                obs_lines.append(json.dumps(item_copy))
            new_observations_block = "[\n  " + ",\n  ".join(obs_lines) + "\n]"

            candidate_map = {}
            embedder = self._get_embedder()
            if embedder is not None:
                for item in items:
                    anchor_text = _obs_anchor_text(item)
                    if not anchor_text:
                        continue
                    vec = self._embed(anchor_text)
                    if vec is None:
                        continue
                    neighbors = self.store.query_similar_nodes(
                        opponent_key, obs_type, vec,
                        top_k=self.faiss_top_k,
                        threshold=self.faiss_edge_threshold,
                    )
                    for n in neighbors[:self.faiss_top_k]:
                        mem = self.store.get_component_memory_for_node(opponent_key, n["obs_id"])
                        if mem:
                            candidate_map[mem["id"]] = mem

            if candidate_map:
                cand_lines = []
                for mem in candidate_map.values():
                    cand_lines.append(_format_memory_for_prompt(mem))
                    cand_lines.append("")
                candidate_memories_block = "\n".join(cand_lines).strip()
            else:
                candidate_memories_block = "(No existing components found — all new observations will start new components.)"

            prompt_template = STEP_A_PROMPTS.get(obs_type, STEP_A_PROMPTS["UNCONDITIONAL"])
            prompt = prompt_template.format(
                n_games=len(all_game_data),
                game_rules=game_rules,
                new_observations_block=new_observations_block,
                candidate_memories=candidate_memories_block,
            )
            msgs = [{"role": "user", "content": prompt}]
            resp, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            raw = resp[0]
            self._log_prompt(f"STEP A: DEDUP + EDGE CREATION ({obs_type})", prompt, raw)

            parsed = extract_json_block(strip_thinking_block(raw))
            return {
                "surviving": parsed.get("surviving_observations", []),
                "edges": parsed.get("edges", [])
            }

        # 3. Run Step A concurrently per type
        all_surviving = []
        all_edges = []
        tasks = [(t, items) for t, items in grouped_items.items() if items]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_obs_type, t, items) for t, items in tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    res = f.result()
                    all_surviving.extend(res["surviving"])
                    all_edges.extend(res["edges"])
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error in Step A for obs type: {e}", exc_info=True)

        # 4. Insert surviving observations into the graph

        # 5. Insert surviving observations into the graph
        changed_component_ids = set()
        temp_id_to_obs_id = {}

        for obs_data in all_surviving:
            temp_id = obs_data.get("id", "")
            obs_type = obs_data.get("type", "")
            count = obs_data.get("count", 1)
            if obs_type not in OBS_TYPES:
                continue

            # Build fields dict by type
            if obs_type == "UNCONDITIONAL":
                fields = {"behavior": obs_data.get("behavior", "")}
            elif obs_type == "CONDITIONAL":
                fields = {
                    "game_state": obs_data.get("game_state", ""),
                    "action": obs_data.get("action", ""),
                }
            elif obs_type == "INFERENCE":
                fields = {
                    "public_action": obs_data.get("public_action", ""),
                    "underlying_driver": obs_data.get("underlying_driver", ""),
                }
            else:
                continue

            # Embed the anchor field for future retrieval
            anchor_text = _obs_anchor_text({"type": obs_type, **fields})
            embedding = self._embed(anchor_text)

            obs_id = self.store.add_observation_node(
                opponent_key, obs_type, fields, embedding=embedding, count=count
            )
            temp_id_to_obs_id[temp_id] = obs_id

        # 6. Apply edges: separate new→existing and new→new
        new_to_existing: dict = {}   # obs_id -> mem_id (existing component)
        new_to_new_edges: list = []  # (obs_id_a, obs_id_b) pairs

        for edge in all_edges:
            source_temp = edge.get("source", "")
            target = edge.get("target", "")
            obs_id = temp_id_to_obs_id.get(source_temp)
            if not obs_id:
                continue

            if target.startswith("new_"):
                # new→new edge: connect two new observations
                target_obs_id = temp_id_to_obs_id.get(target)
                if target_obs_id and obs_id != target_obs_id:
                    new_to_new_edges.append((obs_id, target_obs_id))
            elif target.startswith("mem_"):
                # new→existing component edge
                new_to_existing[obs_id] = target
                self.store.assign_obs_to_component(opponent_key, [obs_id], target)
                changed_component_ids.add(target)

        # 7. Union-Find: cluster new observations connected to each other
        #    so they get synthesized together into one new component
        all_new_obs_ids = list(temp_id_to_obs_id.values())

        parent = {oid: oid for oid in all_new_obs_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in new_to_new_edges:
            if a in parent and b in parent:
                union(a, b)

        # Group clusters: root -> list of obs_ids in that cluster
        clusters: dict = {}
        for obs_id in all_new_obs_ids:
            root = find(obs_id)
            clusters.setdefault(root, []).append(obs_id)

        # 8. Build synthesis_tasks directly (no sentinel strings)
        #    Each task: (mem_id_or_None, new_obs_ids, component_type)
        synthesis_tasks = []
        
        # Track which existing memories got new observations
        mem_new_obs = {}
        for oid, mid in new_to_existing.items():
            mem_new_obs.setdefault(mid, []).append(oid)

        for root, cluster_obs_ids in clusters.items():
            # Check if any node in this cluster is already assigned to existing component
            existing_mem_id = None
            for oid in cluster_obs_ids:
                if oid in new_to_existing:
                    existing_mem_id = new_to_existing[oid]
                    break

            if existing_mem_id:
                # Attach whole cluster to the existing component
                unassigned = [oid for oid in cluster_obs_ids if oid not in new_to_existing]
                if unassigned:
                    self.store.assign_obs_to_component(opponent_key, unassigned, existing_mem_id)
                    mem_new_obs.setdefault(existing_mem_id, []).extend(unassigned)
            else:
                # All new — form a new component
                first_node = self.store.get_observation_node(opponent_key, cluster_obs_ids[0]) if cluster_obs_ids else None
                if first_node:
                    synthesis_tasks.append((None, cluster_obs_ids, first_node["type"]))

        # Add synthesis tasks for all existing components that received new observations
        for mem_id, new_obs_ids in mem_new_obs.items():
            mem = self.store.get_memory(opponent_key, mem_id)
            if mem:
                # Pass ONLY the new observation IDs to Step B, so it updates incrementally
                synthesis_tasks.append((mem_id, new_obs_ids, mem.get("type", "UNCONDITIONAL")))

        # 9. Run Step B with direct task list
        self._run_step_B(opponent_key, synthesis_tasks, game_rules)

    # ------------------------------------------------------------------
    # Step B: Memory Synthesis per changed component
    # ------------------------------------------------------------------

    def _run_step_B(self, opponent_key: str,
                    synthesis_tasks: list,
                    game_rules: str):
        """
        Synthesizes memory entries for a list of component tasks.

        synthesis_tasks: list of (mem_id_or_None, obs_ids, component_type)
          - mem_id is None  → create a new memory entry for these obs_ids
          - mem_id is set   → update the existing memory entry
        """
        if not synthesis_tasks:
            return

        def _synthesize_component(task):
            mem_id, obs_ids, component_type = task

            existing_mem = self.store.get_memory(opponent_key, mem_id) if mem_id else None
            existing_mem_str = (
                _format_memory_for_prompt(existing_mem)
                if existing_mem
                else "(None — creating new memory entry.)"
            )

            nodes_str = self.store.format_component_for_synthesis(opponent_key, mem_id, obs_ids)

            prompt_template = STEP_B_PROMPTS.get(component_type, STEP_B_PROMPTS["UNCONDITIONAL"])
            prompt = prompt_template.format(
                game_rules=game_rules,
                component_type=component_type,
                component_nodes=nodes_str,
                existing_memory=existing_mem_str,
            )
            msgs = [{"role": "user", "content": prompt}]
            resp, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            raw = resp[0]
            self._log_prompt(f"STEP B: MEMORY SYNTHESIS (component={mem_id or 'NEW'})", prompt, raw)

            parsed = extract_json_block(strip_thinking_block(raw))
            if not parsed:
                return

            bullets = parsed.get("bullets", [])
            clean_bullets = [
                {
                    "id": str(b["id"]),
                    "description": str(b["description"]),
                    "count": int(b.get("count", 1)),
                }
                for b in bullets
                if isinstance(b, dict) and "id" in b and "description" in b
            ]

            upserted_mem_id = self.store.upsert_memory(
                opp_key=opponent_key,
                mem_id=mem_id,
                mem_type=component_type,
                name=parsed.get("name", "Unnamed Memory"),
                description=parsed.get("description", ""),
                anchor=parsed.get("anchor", ""),
                bullets=clean_bullets,
                observation_ids=obs_ids,
            )
            self.store.assign_obs_to_component(opponent_key, obs_ids, upserted_mem_id)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_synthesize_component, task) for task in synthesis_tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Step B synthesis error: {e}")

    # ------------------------------------------------------------------
    # Step C: Programmatic consolidation (no LLM call)
    # ------------------------------------------------------------------

    def _run_step_C(self, opponent_key: str, game_rules: str = ""):
        """
        Builds the injected profile block by sorting memories by total
        evidence (sum of bullet counts) and concatenating them.

        Sections order: UNCONDITIONAL → CONDITIONAL → INFERENCE.
        Within each section, memories are sorted by total count descending.
        """
        memories = self.store.get_all_memories(opponent_key)
        if not memories:
            return

        def _total_count(mem: dict) -> int:
            return sum(b.get("count", 0) for b in mem.get("bullets", []))

        section_order = {"UNCONDITIONAL": 0, "CONDITIONAL": 1, "INFERENCE": 2}
        memories_sorted = sorted(
            memories,
            key=lambda m: (section_order.get(m.get("type", ""), 9), -_total_count(m))
        )

        section_labels = {
            "UNCONDITIONAL": "General Tendencies",
            "CONDITIONAL": "Conditional Behaviors",
            "INFERENCE": "Inferred Private Information / Intentions",
        }

        # Group by type and render
        current_section = None
        lines = []
        for mem in memories_sorted:
            mem_type = mem.get("type", "UNCONDITIONAL")
            if mem_type != current_section:
                current_section = mem_type
                label = section_labels.get(mem_type, mem_type)
                lines.append(f"\n--- {label} ---")

            lines.append(f"\n[{mem['id']}] {mem.get('name', 'Unnamed')}")
            lines.append(f"  {mem.get('description', '')}")

            anchor = mem.get("anchor", "")
            if anchor:
                anchor_label = "Game State" if mem_type == "CONDITIONAL" else "Public Action"
                lines.append(f"  {anchor_label}: {anchor}")

            bullets = mem.get("bullets", [])
            total = sum(b.get("count", 0) for b in bullets)
            if bullets:
                if mem_type == "UNCONDITIONAL":
                    bullet_header = "  Observed Behavioral Variants:"
                elif mem_type == "CONDITIONAL":
                    bullet_header = "  Observed Reactions (given the trigger above):"
                else:  # INFERENCE
                    bullet_header = "  Possible Hidden Drivers (inferred from the public action above):"
                lines.append(bullet_header)
                for b in bullets:
                    pct = f"{b['count']}/{total}" if total else "0/0"
                    lines.append(f"    [{b['id']}] {b['description']} ({pct})")
            else:
                lines.append("    (No observations recorded yet.)")

        block = "\n".join(lines).strip()
        self.store.set_consolidated_block(opponent_key, block)

    # ------------------------------------------------------------------
    # Confirmation application (programmatic)
    # ------------------------------------------------------------------

    def _apply_confirmations(self, opponent_key: str, confirmations: list):
        """
        Increments bullet counts based on the post-game confirmation call output.
        Also adds new pending bullets from new_behavior fields.
        Called immediately after the confirmation LLM call, per-game.
        """
        if not confirmations:
            return
        for c in confirmations:
            if not isinstance(c, dict):
                continue
            mem_id = c.get("mem_id", "")
            if not mem_id:
                continue
            for bullet_id in c.get("matched_bullets", []):
                self.store.increment_bullet_count(opponent_key, mem_id, str(bullet_id))
            new_behavior = c.get("new_behavior")
            if new_behavior:
                self.store.add_bullet_to_memory(opponent_key, mem_id, new_behavior)

    # ------------------------------------------------------------------
    # Helper: collect changed component IDs
    # ------------------------------------------------------------------

    def _get_changed_component_ids(self, opponent_key: str) -> set:
        """Returns all component IDs that have observation nodes (for Step B sync)."""
        mem_ids = {m["id"] for m in self.store.get_all_memories(opponent_key)}
        return mem_ids
