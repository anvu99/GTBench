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
    STEP_A_PROMPTS, STEP_B_PROMPTS, STEP_D_PROMPTS,
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
                    use_flash_attn=False,
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
        """Runs Steps A→B→D→C for one opponent."""
        self._run_step_A(opponent_key, all_game_data, game_rules)  # Step A calls Step B
        self._run_step_D(opponent_key, game_rules)                 # Dedup across components
        self._run_step_C(opponent_key, game_rules)                 # Rebuild injected block

    # ------------------------------------------------------------------
    # Step A: Dedup + Edge Creation
    # ------------------------------------------------------------------

    def _run_step_A(self, opponent_key: str, all_game_data: list, game_rules: str):
        """
        Step A: Dedup + Edge Creation.

        For each observation type:
          1. Embed each new raw observation's anchor field.
          2. Query FAISS directly on existing memory anchor embeddings to find
             candidate memories (single-hop: no intermediate observation_nodes).
          3. Call LLM to deduplicate new observations and assign each surviving
             observation to either an existing memory or a new component.
          4. Build synthesis_tasks from the LLM-assigned edges.
          5. Call Step B.
        """
        # 1. Collect all structured observation items from the batch
        structured_items = []
        for game_data in all_game_data:
            for obs in game_data.get("structured_obs", []):
                structured_items.append(obs)

        if not structured_items:
            return

        # 2. Assign temp IDs and group by type
        grouped_items = {t: [] for t in OBS_TYPES}
        for i, item in enumerate(structured_items):
            temp_id = f"new_{i+1:03d}"
            item["temp_id"] = temp_id
            obs_type = item.get("type")
            if obs_type in grouped_items:
                grouped_items[obs_type].append(item)

        def _process_obs_type(obs_type: str, items: list) -> list:
            """
            Runs Step A for one observation type.
            Returns a list of synthesis_tasks: (mem_id_or_None, raw_obs_list, obs_type).
            """
            # Build the new-observations block for the prompt
            obs_lines = []
            for item in items:
                item_copy = {k: v for k, v in item.items() if k != "temp_id"}
                item_copy["id"] = item["temp_id"]
                obs_lines.append(json.dumps(item_copy))
            new_observations_block = "[\n  " + ",\n  ".join(obs_lines) + "\n]"

            # Query FAISS on memory anchor embeddings (single-hop retrieval)
            candidate_map = {}  # mem_id -> memory dict
            embedder = self._get_embedder()
            if embedder is not None:
                for item in items:
                    anchor_text = _obs_anchor_text(item)
                    if not anchor_text:
                        continue
                    vec = self._embed(anchor_text)
                    if vec is None:
                        continue
                    neighbors = self.store.query_similar_memories(
                        opponent_key, obs_type, vec,
                        top_k=self.faiss_top_k,
                        threshold=self.faiss_edge_threshold,
                    )
                    for n in neighbors:
                        candidate_map[n["mem_id"]] = n["memory"]

            if candidate_map:
                cand_lines = []
                for mem in candidate_map.values():
                    cand_lines.append(_format_memory_for_prompt(mem))
                    cand_lines.append("")
                candidate_memories_block = "\n".join(cand_lines).strip()
            else:
                candidate_memories_block = (
                    "(No existing components found — all new observations will start new components.)"
                )

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
            surviving = parsed.get("surviving_observations", [])
            edges = parsed.get("edges", [])

            # --- Build synthesis tasks from LLM output ---
            # Index surviving observations by their temp id
            surviving_by_id = {obs.get("id", ""): obs for obs in surviving}
            items_by_temp_id = {item["temp_id"]: item for item in items}

            # Build edge map: temp_id -> ("existing", mem_id) | ("new", target_temp_id)
            edge_map = {}
            for edge in edges:
                source = edge.get("source", "")
                target = edge.get("target", "")
                if source not in edge_map:
                    if target.startswith("mem_"):
                        edge_map[source] = ("existing", target)
                    elif target.startswith("new_") and target in surviving_by_id:
                        edge_map[source] = ("new", target)

            # Union-Find among surviving temp ids to group new->new clusters
            surv_ids = list(surviving_by_id.keys())
            parent = {sid: sid for sid in surv_ids}

            def find(x):
                while parent.get(x, x) != x:
                    parent[x] = parent.get(parent[x], parent[x])
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for sid in surv_ids:
                if sid in edge_map and edge_map[sid][0] == "new":
                    union(sid, edge_map[sid][1])

            # Group by root
            groups = {}
            for sid in surv_ids:
                root = find(sid)
                groups.setdefault(root, []).append(sid)

            # One synthesis task per group
            tasks = []
            for root, group_ids in groups.items():
                # Determine target memory: prefer an existing mem edge if any in group
                target_mem_id = None
                for gid in group_ids:
                    if gid in edge_map and edge_map[gid][0] == "existing":
                        target_mem_id = edge_map[gid][1]
                        break

                # Build raw obs list from surviving observation data (already LLM-merged)
                raw_obs_list = []
                for gid in group_ids:
                    surv_obs = dict(surviving_by_id[gid])
                    surv_obs.setdefault("count", 1)
                    surv_obs["type"] = obs_type
                    raw_obs_list.append(surv_obs)

                tasks.append((target_mem_id, raw_obs_list, obs_type))

            return tasks

        # 3. Run Step A concurrently per obs type, collect all synthesis tasks
        all_synthesis_tasks = []
        type_tasks = [(t, items) for t, items in grouped_items.items() if items]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_obs_type, t, items) for t, items in type_tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    all_synthesis_tasks.extend(f.result())
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error in Step A: {e}", exc_info=True)

        # 4. Run Step B with the collected synthesis tasks
        self._run_step_B(opponent_key, all_synthesis_tasks, game_rules)

    # ------------------------------------------------------------------
    # Step B: Memory Synthesis per component
    # ------------------------------------------------------------------

    def _run_step_B(self, opponent_key: str,
                    synthesis_tasks: list,
                    game_rules: str):
        """
        Synthesizes memory entries for a list of component tasks.

        synthesis_tasks: list of (mem_id_or_None, raw_obs_list, component_type)
          - mem_id is None     → create a new memory entry
          - mem_id is a str    → update the existing memory entry
          - raw_obs_list       → list of raw observation dicts (already LLM-deduped by Step A)

        After synthesis, the anchor field is embedded and stored so FAISS stays current.
        """
        if not synthesis_tasks:
            return

        def _synthesize_component(task):
            mem_id, raw_obs_list, component_type = task

            existing_mem = self.store.get_memory(opponent_key, mem_id) if mem_id else None
            existing_mem_str = (
                _format_memory_for_prompt(existing_mem)
                if existing_mem
                else "(None — creating new memory entry.)"
            )

            nodes_str = self.store.format_raw_obs_for_synthesis(raw_obs_list)

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

            # Embed the memory for FAISS retrieval in future batches.
            # UNCONDITIONAL memories have no anchor (it's always ""), so we embed
            # the concatenation of bullet descriptions as the best proxy for
            # matching future UNCONDITIONAL observations (which embed their behavior field).
            anchor = parsed.get("anchor", "")
            if component_type == "UNCONDITIONAL":
                bullet_texts = " | ".join(
                    b.get("description", "") for b in clean_bullets[:6]
                )
                embed_text = bullet_texts or parsed.get("description", "")
            else:
                embed_text = anchor  # CONDITIONAL: game_state; INFERENCE: public_action
            embedding = self._embed(embed_text) if embed_text.strip() else None

            self.store.upsert_memory(
                opp_key=opponent_key,
                mem_id=mem_id,
                mem_type=component_type,
                name=parsed.get("name", "Unnamed Memory"),
                description=parsed.get("description", ""),
                anchor=anchor,
                bullets=clean_bullets,
                embedding=embedding,
                rebuild_faiss=False,  # defer FAISS rebuild to avoid concurrent-rebuild race
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_synthesize_component, task) for task in synthesis_tasks]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Step B synthesis error: {e}")

        # Rebuild FAISS once per obs_type after ALL concurrent upserts are done.
        # This avoids the race where concurrent _rebuild_faiss_index calls snapshot
        # different subsets of memories and the last-to-finish overwrites with a
        # partial index.
        for obs_type in OBS_TYPES:
            self.store._rebuild_faiss_index(opponent_key, obs_type)

    # ------------------------------------------------------------------
    # Step D: Cross-component deduplication
    # ------------------------------------------------------------------

    def _run_step_D(self, opponent_key: str, game_rules: str):
        """
        Step D: Cross-component deduplication (runs after Step B, before Step C).

        Fires 3 concurrent LLM calls — one per observation type. Each call
        receives ALL existing memories of that type and is asked to identify
        duplicates. The backend applies the merge operations programmatically:
          - upsert the kept component with merged bullets
          - delete the discarded component(s)
          - FAISS index is rebuilt automatically by upsert/delete
        """
        def _dedup_type(obs_type: str):
            all_memories_block = self.store.format_memories_by_type_for_dedup(
                opponent_key, obs_type
            )
            # Skip if only 0 or 1 component — nothing to deduplicate
            mems_of_type = [
                m for m in self.store.get_all_memories(opponent_key)
                if m.get("type") == obs_type
            ]
            if len(mems_of_type) <= 1:
                return

            prompt_template = STEP_D_PROMPTS[obs_type]
            prompt = prompt_template.format(
                game_rules=game_rules,
                all_memories_block=all_memories_block,
            )
            msgs = [{"role": "user", "content": prompt}]
            resp, _ = self.llm_query(msgs, n=1, stop=None, prompt_type='move')
            raw = resp[0]
            self._log_prompt(f"STEP D: DEDUPLICATION ({obs_type})", prompt, raw)

            parsed = extract_json_block(strip_thinking_block(raw))
            if not parsed:
                return

            for group in parsed.get("groups", []):
                keep_id = group.get("keep_id", "")
                discard_ids = group.get("discard_ids", [])
                updated_bullets_raw = group.get("updated_bullets", [])
                if not keep_id or not discard_ids:
                    continue
                # Validate that keep_id and all discard_ids actually exist
                if not self.store.get_memory(opponent_key, keep_id):
                    continue
                clean_bullets = [
                    {
                        "id": str(b["id"]),
                        "description": str(b["description"]),
                        "count": int(b.get("count", 1)),
                    }
                    for b in updated_bullets_raw
                    if isinstance(b, dict) and "id" in b and "description" in b
                ]
                # Compute embedding for the kept (merged) memory
                updated_anchor = group.get("updated_anchor", "")
                if obs_type == "UNCONDITIONAL":
                    bullet_texts = " | ".join(
                        b.get("description", "") for b in clean_bullets[:6]
                    )
                    embed_text = bullet_texts or group.get("updated_description", "")
                else:
                    embed_text = updated_anchor
                embedding = self._embed(embed_text) if embed_text.strip() else None

                # Upsert the kept memory with merged content
                self.store.upsert_memory(
                    opp_key=opponent_key,
                    mem_id=keep_id,
                    mem_type=obs_type,
                    name=group.get("updated_name", ""),
                    description=group.get("updated_description", ""),
                    anchor=updated_anchor,
                    bullets=clean_bullets,
                    embedding=embedding,
                )
                # Delete each discarded component
                for discard_id in discard_ids:
                    if discard_id != keep_id:  # safety guard
                        self.store.delete_memory(opponent_key, discard_id)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(_dedup_type, t) for t in OBS_TYPES]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Step D dedup error: {e}")


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
