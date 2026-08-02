import copy
import logging
import os.path
import argparse
import pathlib
import queue
import threading
from gamingbench.utils import utils
from gamingbench.environments.base_env import BaseGameEnv
import json

games = ['tictactoe', 'connect4', 'texasholdem', 'neuron_poker', 'backgammon', 'breakthrough',
         'first_sealed_auction', 'gin_rummy', 'liars_dice', 'negotiation', 'nim', 'pig', 'kuhn_poker',
         'prisoners_dilemma', 'cooperative_negotiation', 'hanabi', 'hanabi-micro', 'hanabi-small', 'hanabi3-micro', 'hanabi-small-custom']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-matches', type=int,
                        default=100, help='The number gaming matches')
    parser.add_argument('--exp-root', type=str, default='../experiments')
    parser.add_argument('--seed', type=int, default=0)
    # gaming parameters
    parser.add_argument('--game-names', type=str, nargs='+',
                        default=['tictactoe'], choices=games)
    parser.add_argument('--game-config-root', type=str, default='gamingbench/configs/game_configs',
                        help='Path of gaming environment configurations')
    # agent parameters
    parser.add_argument('--agent-configs', type=str, nargs='+',
                        default=[
                        ],
                        help='List of paths of agent configurations')

    parser.add_argument('--model-configs', type=str, nargs='+',
                        default=[
                        ],
                        help='List of paths of model configurations')

    parser.add_argument('--output-folder', default='./')

    parser.add_argument('--api-keys', default='', nargs='+')

    parser.add_argument('--exchange-first-player',
                        default=False, action='store_true')
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--threshold-matches', default=50, type=int)
    parser.add_argument('--enable-chat', default=False, action='store_true', help='Enable chat feature for agents')
    parser.add_argument('--think-further', default=False, action='store_true', help='Instruct agent to think multiple steps ahead')
    parser.add_argument('--n-player-memory-mode', type=str, default='combined', choices=['combined', 'separate'], help='Memory mode for N-player cooperative games')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of games per LTM update batch. Default 1 = update after every game (existing behaviour).')
    args = parser.parse_args()

    return args


def _get_agent_store_path(agent):
    """Helper to extract the correct store path for any memory agent."""
    return getattr(agent, 'ew_store_path', 
           getattr(agent, 'sw_store_path', 
           getattr(agent, 'ltm_store_path', 
           getattr(agent, 'rules_store_path', 
           getattr(agent, 'memory_bank_path', 
           getattr(agent, 'store_path', None))))))

def _get_memory_snapshot(agent):
    """Returns a serializable snapshot of the agent's memory."""
    if hasattr(agent, 'sw_store'):
        return {'sw': dict(agent.sw_store.store)}
    elif hasattr(agent, 'ew_store'):
        return {
            'ew_observations': {k: list(v) for k, v in agent.ew_store.observations.items()},
            'ew_notes': dict(agent.ew_store.notes)
        }
    elif hasattr(agent, 'ltm_store'):
        snap = {
            'ltm': dict(agent.ltm_store.store)
        }
        if hasattr(agent, 'self_ltm_store'):
            snap['self_ltm'] = dict(agent.self_ltm_store.store)
        if hasattr(agent, 'proactive_ltm_store'):
            snap['proactive_ltm'] = dict(agent.proactive_ltm_store.store)
        if hasattr(agent, 'partner_ltm_store') and hasattr(agent, 'partner_ltm_store_path') and agent.partner_ltm_store_path:
            if os.path.exists(agent.partner_ltm_store_path):
                agent.partner_ltm_store.load(agent.partner_ltm_store_path)
            snap['partner_ltm'] = dict(agent.partner_ltm_store.store)
        return snap
    elif hasattr(agent, 'opp_store') and hasattr(agent, 'self_store') and hasattr(agent, 'proac_store'):
        return {
            'opp': copy.deepcopy(agent.opp_store.store),
            'self': copy.deepcopy(agent.self_store.store),
            'proac': copy.deepcopy(agent.proac_store.store),
            'opp_gy': copy.deepcopy(agent.opp_store.graveyard),
            'self_gy': copy.deepcopy(agent.self_store.graveyard),
            'proac_gy': copy.deepcopy(agent.proac_store.graveyard),
        }
    elif hasattr(agent, 'rules') and hasattr(agent, 'experience_pool'):
        return {
            'rules': copy.deepcopy(agent.rules),
            'experience_pool': copy.deepcopy(agent.experience_pool)
        }
    elif hasattr(agent, 'memory_bank'):
        return {
            'memory_bank': copy.deepcopy(agent.memory_bank)
        }
    elif hasattr(agent, 'store') and hasattr(agent.store, 'evidence'):
        return {
            'evidence': copy.deepcopy(agent.store.evidence),
            'memories': copy.deepcopy(agent.store.memories)
        }
    return {}

def _restore_memory_snapshot(clone, snapshot):
    """Restores memory from a snapshot and prepares for batch mode."""
    if hasattr(clone, 'sw_store'):
        if 'sw' in snapshot:
            clone.sw_store.store = dict(snapshot['sw'])
        clone.sw_store_path = '/dev/null'
    if hasattr(clone, 'ew_store'):
        from collections import deque
        if 'ew_observations' in snapshot:
            clone.ew_store.observations = {k: deque(v) for k, v in snapshot['ew_observations'].items()}
        if 'ew_notes' in snapshot:
            clone.ew_store.notes = dict(snapshot['ew_notes'])
        clone.ew_store_path = '/dev/null'
    if hasattr(clone, 'ltm_store'):
        if 'ltm' in snapshot:
            clone.ltm_store.store = dict(snapshot['ltm'])
        if hasattr(clone, 'self_ltm_store'):
            clone.self_ltm_store.store = dict(snapshot.get('self_ltm', {}))
        if 'proactive_ltm' in snapshot and hasattr(clone, 'proactive_ltm_store'):
            clone.proactive_ltm_store.store = dict(snapshot['proactive_ltm'])
            clone.proactive_ltm_store_path = '/dev/null'
        if 'partner_ltm' in snapshot and hasattr(clone, 'partner_ltm_store'):
            clone.partner_ltm_store.store = dict(snapshot['partner_ltm'])
            clone.partner_ltm_store_path = '/dev/null'
        clone.ltm_store_path = '/dev/null'
        clone.self_ltm_store_path = '/dev/null'
    if hasattr(clone, 'opp_store') and hasattr(clone, 'self_store') and hasattr(clone, 'proac_store'):
        if 'opp' in snapshot:
            clone.opp_store.store = copy.deepcopy(snapshot['opp'])
            clone.opp_store.graveyard = copy.deepcopy(snapshot.get('opp_gy', {}))
        if 'self' in snapshot:
            clone.self_store.store = copy.deepcopy(snapshot['self'])
            clone.self_store.graveyard = copy.deepcopy(snapshot.get('self_gy', {}))
        if 'proac' in snapshot:
            clone.proac_store.store = copy.deepcopy(snapshot['proac'])
            clone.proac_store.graveyard = copy.deepcopy(snapshot.get('proac_gy', {}))
        clone.opp_store_path = '/dev/null'
        clone.self_store_path = '/dev/null'
        clone.proac_store_path = '/dev/null'
    if hasattr(clone, 'rules') and hasattr(clone, 'experience_pool'):
        if 'rules' in snapshot:
            clone.rules = copy.deepcopy(snapshot['rules'])
        if 'experience_pool' in snapshot:
            clone.experience_pool = copy.deepcopy(snapshot['experience_pool'])
    if hasattr(clone, 'memory_bank'):
        if 'memory_bank' in snapshot:
            clone.memory_bank = copy.deepcopy(snapshot['memory_bank'])
        clone.memory_bank_path = '/dev/null'
    if hasattr(clone, 'store') and hasattr(clone.store, 'evidence'):
        if 'evidence' in snapshot:
            clone.store.evidence = copy.deepcopy(snapshot['evidence'])
            clone.store.memories = copy.deepcopy(snapshot['memories'])
        clone.store_path = '/dev/null'

def clone_agent_for_batch(original_agent, memory_snapshot: dict):
    """Create an independent agent copy seeded with a frozen memory snapshot.

    The clone:
    - Has its own in-memory store pre-loaded from memory_snapshot (no file I/O during game).
    - Has batch_mode=True so post_game_update() stores gradient data and returns early.
    - Does NOT call set_storage_dir() — it never writes to disk (store_path='/dev/null').
    """
    clone = copy.deepcopy(original_agent)
    clone._parent_store_path = _get_agent_store_path(original_agent)
    
    if hasattr(original_agent, 'self_store_path'):
        clone._parent_self_store_path = original_agent.self_store_path
    if hasattr(original_agent, 'proac_store_path'):
        clone._parent_proac_store_path = original_agent.proac_store_path
    if hasattr(original_agent, 'self_ltm_store_path'):
        clone._parent_self_ltm_store_path = original_agent.self_ltm_store_path
    if hasattr(original_agent, 'proactive_ltm_store_path'):
        clone._parent_proactive_ltm_store_path = original_agent.proactive_ltm_store_path
        
    _restore_memory_snapshot(clone, memory_snapshot)
    clone.batch_mode = True
    clone._last_batch_result = None
    return clone

def run_game(game_name):
    game_config = utils.load_config(os.path.join(args.game_config_root, f'{game_name}.yaml'))
    if getattr(game_config, 'num_players', 2) > 2 or len(args.agent_configs) > 2:
        return run_game_nplayer(game_name)

    log_root = args.exp_root
    pathlib.Path(log_root).mkdir(parents=True, exist_ok=True)
    agent_names = [a.split('/')[-1].split('.')[0] for a in args.agent_configs]
    model_names = [m.split('/')[-1].split('.')[0] for m in args.model_configs]

    run_name = f'{agent_names[0]}_{model_names[0]}_{agent_names[1]}_{model_names[1]}'

    log_path = os.path.join(log_root, run_name + '.log')
    logger = utils.LLMBenchLogger(log_path)
    result_path = os.path.join(log_root, run_name + '.jsonl')

    if not os.path.exists(result_path):
        file = open(result_path, 'w')
        file.close()

    # initialize env and game
    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(
        args.game_config_root, f'{game_name}.yaml'))
    game_env.save_game_config(utils.load_config(
        os.path.join(args.game_config_root, f'{game_name}.yaml')))

    # initialize agents
    agents = [utils.load_agent(config_path, game=game.game, game_config=game.config)
              for config_path in args.agent_configs]
    models = [utils.load_model(config_path)
              for config_path in args.model_configs]

    for i, (a, m) in enumerate(zip(agents, models)):
        a.set_model(m)
        a.player_id = f"p{i}"
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        a.memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
        if hasattr(a, 'set_storage_dir'):
            a.set_storage_dir(log_root)

    if len(agents) == 2:
        a0, a1 = agents[0], agents[1]
        if (getattr(a0, 'hive_mode', False) and hasattr(a0, 'set_partner_store')
                and getattr(a1, 'hive_mode', False) and hasattr(a1, 'set_partner_store')):
            if hasattr(a0, 'agent_name') and hasattr(a1, 'agent_name') and a0.agent_name == a1.agent_name:
                key_a0 = a0.agent_name
                key_a1 = a1.agent_name
            else:
                key_a0 = f"{a0.player_id}:{a0.agent_name}_{models[0].nick_name}"
                key_a1 = f"{a1.player_id}:{a1.agent_name}_{models[1].nick_name}"
            a0.set_partner_store(a1.ltm_store_path, key_a0)
            a1.set_partner_store(a0.ltm_store_path, key_a1)



    for config_path in args.model_configs:
        game_env.append_models_config(utils.load_config(config_path))

    game_env.set_game(game)

    lock = threading.Lock()
    batch_size = getattr(args, 'batch_size', 1)

    if batch_size > 1:
        # ── Batch mode ────────────────────────────────────────────────────────
        # Games within a batch run in parallel; batches are sequential.
        # Each game gets its own independent agent clone pre-seeded with the
        # same LTM snapshot — no shared mutable state across parallel games.
        #
        # Deduplicate by ltm_store_path: agents and reversed_agents typically
        # contain 2 LTMAgent instances with the same path. Only flush through
        # one to avoid applying the same gradients twice.
        seen_store_paths = set()
        batch_agents = []
        for a in agents:
            if hasattr(a, 'flush_batch_updates'):
                store_path = _get_agent_store_path(a)
                if store_path and store_path not in seen_store_paths:
                    seen_store_paths.add(store_path)
                    batch_agents.append(a)
        results = []
        for batch_start in range(0, args.num_matches, batch_size):
            batch_end = min(batch_start + batch_size, args.num_matches)
            batch_q = queue.Queue()  # thread-safe gradient data collector

            # Freeze the current memory store state for all games in this batch
            memory_snapshots = {
                _get_agent_store_path(a): _get_memory_snapshot(a) 
                for a in batch_agents
            }

            # Build per-game args, each with its own agent clones
            batch_args = []
            for match_idx in range(batch_start, batch_end):
                fresh_agents = []
                for a in agents:
                    if hasattr(a, 'flush_batch_updates'):
                        store_path = _get_agent_store_path(a)
                        snap = memory_snapshots.get(store_path, _get_memory_snapshot(a))
                        fresh_agents.append(clone_agent_for_batch(a, snap))
                    else:
                        fresh_agents.append(copy.deepcopy(a))

                for fa, m in zip(fresh_agents, models):
                    fa.set_model(m)
                    fa.enable_chat = getattr(args, 'enable_chat', False)
                    fa.think_further = getattr(args, 'think_further', False)

                batch_args.append({
                    'match_idx': match_idx,
                    'game_name': game_name,
                    'agents': fresh_agents,
                    'models': models,
                    'result_path': result_path,
                    'args': args,
                    'lock': lock,
                    'batch_queue': batch_q,
                })

            # Run all N games in the batch in parallel; blocks until all complete
            batch_results = utils.parallel_func(
                run_match, batch_args,
                num_workers=min(args.num_workers, len(batch_args))
            )
            results.extend(batch_results)

            # Collect gradient data deposited by each game's agent
            gradient_data = []
            while not batch_q.empty():
                gradient_data.append(batch_q.get())

            # Flush: single unified synthesis + EMA for all N games
            if gradient_data and batch_agents:
                # Borrow opponent context (current_game_intro) from the first successful clone
                # that actually played to pass it to the base agent for context.
                for batch_agent in batch_agents:
                    if not getattr(batch_agent, 'current_game_intro', None):
                        sample = next(
                            (a for ba in batch_args for a in ba['agents'] if hasattr(a, 'current_game_intro') and a.current_game_intro),
                            None
                        )
                        if sample:
                            for batch_agent_inner in batch_agents:
                                batch_agent_inner.current_game_intro  = sample.current_game_intro
                                batch_agent_inner.current_game_name = game_name
                            break
                from gamingbench.prompts.observation_prompts import construct_game_intro
                flushed_paths = set()
                for batch_agent in batch_agents:
                    batch_agent.current_game_name = game_name.lower()
                    if not getattr(batch_agent, 'current_game_intro', None):
                        batch_agent.current_game_intro = construct_game_intro(game_name, game_config=game.config)
                    store_path = _get_agent_store_path(batch_agent)
                    
                    if store_path in flushed_paths:
                        continue
                    
                    agent_data = [d for p_path, d in gradient_data if p_path == store_path]
                    if agent_data:
                        flushed_paths.add(store_path)
                        batch_agent.flush_batch_updates(agent_data)

        results = [r[0] for r in results]

    elif args.num_workers == 1:
        results = []
        for match_idx in range(args.num_matches):
            match_arg = {
                'match_idx': match_idx,
                'game_name': game_name,
                'agents': [copy.deepcopy(a) for a in agents],
                'models': models,
                'result_path': result_path,
                'args': args,
                'lock': lock
            }
            results.append(run_match(match_arg))
    else:
        match_arg_list = []
        for match_idx in range(args.num_matches):
            match_arg_list.append({
                'match_idx': match_idx,
                'game_name': game_name,
                'models': models,
                'agents': [copy.deepcopy(a) for a in agents],
                'result_path': result_path,
                'args': args,
                'lock': lock
            })
        results = utils.parallel_func(run_match, match_arg_list,
                                      num_workers=args.num_workers)
        remaining_matches_param = pick_out_invalid_matches(results)

        while len(results) < args.threshold_matches and len(remaining_matches_param) != 0:
            added_results = utils.parallel_func(run_match, remaining_matches_param,
                                                num_workers=min(args.num_workers, len(remaining_matches_param)))
            results = results + added_results
            remaining_matches_param = pick_out_invalid_matches(added_results)
        # save to jsonl
        results = [r[0] for r in results]
    # utils.save_jsonl(results, result_path)



def pick_out_invalid_matches(results):
    invalid_matches_param = []
    for history, parameters in results:
        if history["matches"][0]["status"] != "Normal":
            invalid_matches_param.append(parameters)
    return invalid_matches_param


def run_match(params):
    match_idx = params['match_idx']
    game_name = params['game_name']
    agents = params['agents']
    models = params['models']
    result_path = params['result_path']

    args = params['args']
    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(
        args.game_config_root, f'{game_name}.yaml'))
    game_env.save_game_config(utils.load_config(
        os.path.join(args.game_config_root, f'{game_name}.yaml')))

    game_env.set_game(game)

    game_env.set_agents(agents)
    game_env.set_models(models)

    for config_path in args.agent_configs:
        game_env.append_agents_config(utils.load_config(config_path))

    for config_path in args.model_configs:
        game_env.append_models_config(utils.load_config(config_path))

    # ── Per-game log file ────────────────────────────────────────────────────
    # e.g. ltm_agent_gemini-3_prompt_agent_gemini-3_game_0003.log
    run_name = os.path.splitext(os.path.basename(result_path))[0]
    per_game_log = os.path.join(
        os.path.dirname(result_path),
        f'{run_name}_game_{match_idx:04d}.log'
    )
    game_log_handler = utils.add_game_log_handler(per_game_log)

    try:
        first_player = 1 if (args.exchange_first_player and match_idx % 2 == 1) else 0
        game_env.play(first_player=first_player)
        res = game_env.history_tracker.to_dict()
    finally:
        utils.remove_game_log_handler(game_log_handler)

    with params['lock']:
        with open(result_path, 'a') as file:
            file.writelines(json.dumps(res) + '\n')

    # ── Batch mode: deposit gradient data into the shared queue ──────────────
    # Check BOTH agents and reversed_agents: for odd matches (exchange_first_player)
    # the game runs with reversed_agents, so the LTM clone in reversed_agents has
    # the _last_batch_result, not the one in agents.
    batch_queue = params.get('batch_queue')
    if batch_queue is not None:
        all_agents_in_match = list(params.get('agents', []))
        for agent in all_agents_in_match:
            if getattr(agent, 'batch_mode', False) and agent._last_batch_result is not None:
                batch_queue.put((getattr(agent, '_parent_store_path', None), agent._last_batch_result))
                agent._last_batch_result = None

    return (res, params)



def run_match_nplayer(params):
    match_idx = params['match_idx']
    game_name = params['game_name']
    result_path = params['result_path']
    lock = params['lock']
    args = params['args']
    log_root = params['log_root']

    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(args.game_config_root, f'{game_name}.yaml'))
    game_env.save_game_config(utils.load_config(os.path.join(args.game_config_root, f'{game_name}.yaml')))

    agents = params['agents']
    models = params['models']

    # Wait, run_game_nplayer will initialize models and set them on agents!
    # BUT run_match clones agents but sets their models again. Let's keep it safe.
    for a, m in zip(agents, models):
        a.set_model(m)
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)

    # Note: args.agent_configs and args.model_configs might not be correct if we duplicate agents.
    # In batch mode, we don't strictly need to append them to game_env if they are already in game_env,
    # but we will just pass them.
    for a_config in params.get('agent_configs', []):
        game_env.append_agents_config(utils.load_config(a_config))
        
    for m_config in params.get('model_configs', []):
        game_env.append_models_config(utils.load_config(m_config))

    game_env.set_game(game)

    # ── Per-game log file ────────────────────────────────────────────────────
    run_name = os.path.splitext(os.path.basename(result_path))[0]
    per_game_log = os.path.join(
        os.path.dirname(result_path),
        f'{run_name}_game_{match_idx:04d}.log'
    )
    game_log_handler = utils.add_game_log_handler(per_game_log)
    logger = logging.getLogger('gamingbench.utils.utils')
    
    logger.info(f'Game {match_idx} starts')
    for a in agents:
        a.logger = logger
    game_env.logger = logger
    if hasattr(game_env, 'game') and hasattr(game_env.game, 'logger'):
        game_env.game.logger = logger

    # Initialize agent memory tracking
    from gamingbench.prompts.observation_prompts import construct_game_intro
    for i, agent in enumerate(agents):
        if hasattr(agent, 'reset_game_state'):
            game_intro = construct_game_intro(game_name, enable_chat=getattr(agent, 'enable_chat', False), game_config=game_env.game.config)
            opponent_keys = []
            for j, other_agent in enumerate(agents):
                if i != j:
                    if hasattr(agent, 'agent_name') and hasattr(other_agent, 'agent_name') and agent.agent_name == other_agent.agent_name:
                        opponent_keys.append(other_agent.agent_name)
                    else:
                        player_id = getattr(other_agent, 'player_id', f"p{j}")
                        opponent_keys.append(f"{player_id}:{other_agent.agent_name}_{models[j].nick_name}")
            
            memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
            if memory_mode == 'combined':
                agent.reset_game_state("+".join(sorted(opponent_keys)), game_intro)
            else:
                agent.reset_game_state(sorted(opponent_keys), game_intro)

    try:
        from gamingbench.utils.history_tracker import HistoryTracker
        tracker = HistoryTracker()
        game_env.game.play(agents, models, tracker, seat_mapping=params.get('seat_mapping'))
        logger.info(f'Game {match_idx} ends')
        logger.info(tracker.matches[-1].to_dict())

        match_dict = tracker.matches[-1].to_dict()
        match_dict['match_idx'] = match_idx

        with lock:
            with open(result_path, 'a') as f:
                f.write(json.dumps(match_dict) + '\n')

    finally:
        utils.remove_game_log_handler(game_log_handler)
        
    # ── Batch mode: deposit gradient data into the shared queue ──────────────
    batch_queue = params.get('batch_queue')
    if batch_queue is not None:
        all_agents_in_match = list(params.get('agents', []))
        for agent in all_agents_in_match:
            if getattr(agent, 'batch_mode', False) and agent._last_batch_result is not None:
                batch_queue.put((getattr(agent, '_parent_store_path', None), agent._last_batch_result))
                agent._last_batch_result = None

    return match_dict

def run_game_nplayer(game_name):
    game_config = utils.load_config(os.path.join(args.game_config_root, f'{game_name}.yaml'))
    num_players = getattr(game_config, 'num_players', 2)

    agent_configs = list(args.agent_configs)
    model_configs = list(args.model_configs)
    
    # Expand configs if num_players > provided configs
    while len(agent_configs) < num_players:
        agent_configs.append(agent_configs[-1])
    while len(model_configs) < num_players:
        model_configs.append(model_configs[-1])

    log_root = args.exp_root
    pathlib.Path(log_root).mkdir(parents=True, exist_ok=True)
    
    agent_names = [a.split('/')[-1].split('.')[0] for a in agent_configs]
    model_names = [m.split('/')[-1].split('.')[0] for m in model_configs]

    # Run name joins all agents
    parts = []
    for a, m in zip(agent_names, model_names):
        parts.append(f"{a}_{m}")
    run_name = "_".join(parts)

    log_path = os.path.join(log_root, run_name + '.log')
    logger = utils.LLMBenchLogger(log_path)
    result_path = os.path.join(log_root, run_name + '.jsonl')

    if not os.path.exists(result_path):
        with open(result_path, 'w') as file:
            pass

    game = utils.load_game(os.path.join(args.game_config_root, f'{game_name}.yaml'))
    agents = [utils.load_agent(config_path, game=game.game, game_config=game.config) for config_path in agent_configs]
    models = [utils.load_model(config_path) for config_path in model_configs]

    for i, (a, m) in enumerate(zip(agents, models)):
        a.set_model(m)
        a.player_id = f"p{i}"
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        a.memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
        if hasattr(a, 'set_storage_dir'):
            a.set_storage_dir(log_root)

    if len(agents) == 2:
        a0, a1 = agents[0], agents[1]
        if (getattr(a0, 'hive_mode', False) and hasattr(a0, 'set_partner_store')
                and getattr(a1, 'hive_mode', False) and hasattr(a1, 'set_partner_store')):
            if hasattr(a0, 'agent_name') and hasattr(a1, 'agent_name') and a0.agent_name == a1.agent_name:
                key_a0 = a0.agent_name
                key_a1 = a1.agent_name
            else:
                key_a0 = f"{a0.player_id}:{a0.agent_name}_{models[0].nick_name}"
                key_a1 = f"{a1.player_id}:{a1.agent_name}_{models[1].nick_name}"
            a0.set_partner_store(a1.ltm_store_path, key_a0)
            a1.set_partner_store(a0.ltm_store_path, key_a1)

    lock = threading.Lock()
    batch_size = getattr(args, 'batch_size', 1)
    
    if batch_size > 1:
        # ── Batch mode ────────────────────────────────────────────────────────
        seen_store_paths = set()
        batch_agents = []
        for a in agents:
            if hasattr(a, 'flush_batch_updates'):
                store_path = getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', getattr(a, 'memory_bank_path', None)))))
                if store_path and store_path not in seen_store_paths:
                    seen_store_paths.add(store_path)
                    batch_agents.append(a)
        results = []
        for batch_start in range(0, args.num_matches, batch_size):
            batch_end = min(batch_start + batch_size, args.num_matches)
            batch_q = queue.Queue()  # thread-safe gradient data collector

            memory_snapshots = {
                getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', getattr(a, 'memory_bank_path', None))))): _get_memory_snapshot(a) 
                for a in batch_agents
            }

            batch_args = []
            for match_idx in range(batch_start, batch_end):
                fresh_agents = []
                for a in agents:
                    if hasattr(a, 'flush_batch_updates'):
                        store_path = getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', getattr(a, 'memory_bank_path', None)))))
                        snap = memory_snapshots.get(store_path, _get_memory_snapshot(a))
                        fresh_agents.append(clone_agent_for_batch(a, snap))
                    else:
                        fresh_agents.append(copy.deepcopy(a))

                if getattr(args, 'exchange_first_player', False):
                    import itertools
                    perms = list(itertools.permutations(range(num_players)))
                    p = list(perms[match_idx % len(perms)])
                else:
                    p = list(range(num_players))

                match_agents = fresh_agents
                match_models = list(models)
                match_a_configs = list(agent_configs)
                match_m_configs = list(model_configs)

                for fa, m in zip(match_agents, match_models):
                    fa.set_model(m)
                    fa.enable_chat = getattr(args, 'enable_chat', False)
                    fa.think_further = getattr(args, 'think_further', False)

                batch_args.append({
                    'match_idx': match_idx,
                    'game_name': game_name,
                    'agents': match_agents,
                    'models': match_models,
                    'result_path': result_path,
                    'args': args,
                    'lock': lock,
                    'batch_queue': batch_q,
                    'agent_configs': match_a_configs,
                    'model_configs': match_m_configs,
                    'log_root': log_root,
                    'seat_mapping': p
                })

            batch_results = utils.parallel_func(
                run_match_nplayer, batch_args,
                num_workers=min(args.num_workers, len(batch_args))
            )
            results.extend(batch_results)

            gradient_data = []
            while not batch_q.empty():
                gradient_data.append(batch_q.get())

            if gradient_data and batch_agents:
                for batch_agent in batch_agents:
                    if not getattr(batch_agent, 'current_game_intro', None):
                        sample = next(
                            (a for ba in batch_args for a in ba['agents'] if hasattr(a, 'current_game_intro') and a.current_game_intro),
                            None
                        )
                        if sample:
                            for batch_agent_inner in batch_agents:
                                batch_agent_inner.current_game_intro  = sample.current_game_intro
                                batch_agent_inner.current_game_name = game_name
                            break
                from gamingbench.prompts.observation_prompts import construct_game_intro
                flushed_paths = set()
                for batch_agent in batch_agents:
                    batch_agent.current_game_name = game_name.lower()
                    if not getattr(batch_agent, 'current_game_intro', None):
                        # game config is not easily accessible here? We need game.config
                        batch_agent.current_game_intro = construct_game_intro(game_name, game_config=game.config)
                    store_path = getattr(batch_agent, 'ew_store_path', getattr(batch_agent, 'sw_store_path', getattr(batch_agent, 'ltm_store_path', getattr(batch_agent, 'rules_store_path', getattr(batch_agent, 'memory_bank_path', None)))))
                    
                    if store_path in flushed_paths:
                        continue
                        
                    agent_data = [d for p_path, d in gradient_data if p_path == store_path]
                    if agent_data:
                        flushed_paths.add(store_path)
                        batch_agent.flush_batch_updates(agent_data)

    elif args.num_workers == 1:
        for match_idx in range(args.num_matches):
            if getattr(args, 'exchange_first_player', False):
                import itertools
                perms = list(itertools.permutations(range(num_players)))
                p = list(perms[match_idx % len(perms)])
            else:
                p = list(range(num_players))
                
            match_agents = [copy.deepcopy(a) for a in agents]
            match_models = models
            match_a_configs = agent_configs
            match_m_configs = model_configs
                
            match_arg = {
                'match_idx': match_idx,
                'game_name': game_name,
                'agents': match_agents,
                'models': match_models,
                'result_path': result_path,
                'args': args,
                'lock': lock,
                'agent_configs': match_a_configs,
                'model_configs': match_m_configs,
                'log_root': log_root,
                'seat_mapping': p
            }
            run_match_nplayer(match_arg)
    else:
        params_list = []
        for match_idx in range(args.num_matches):
            if getattr(args, 'exchange_first_player', False):
                import itertools
                perms = list(itertools.permutations(range(num_players)))
                p = list(perms[match_idx % len(perms)])
            else:
                p = list(range(num_players))
                
            match_agents = [copy.deepcopy(a) for a in agents]
            match_models = models
            match_a_configs = agent_configs
            match_m_configs = model_configs
                
            params_list.append({
                'match_idx': match_idx,
                'game_name': game_name,
                'agents': match_agents, # Not safe to share directly but legacy behavior if batch=1 and workers>1
                'models': match_models,
                'result_path': result_path,
                'lock': lock,
                'args': args,
                'log_root': log_root,
                'agent_configs': match_a_configs,
                'model_configs': match_m_configs,
                'seat_mapping': p
            })
        utils.parallel_func(run_match_nplayer, params_list, num_workers=min(args.num_workers, len(params_list)))

    with open(result_path, 'r') as f:
        matches = [json.loads(line) for line in f.readlines()]

    scores = [m.get('winner_score', 0) for m in matches]
    if scores:
        avg_score = sum(scores) / len(scores)
        logger.info(f"Average Score: {avg_score}")

    return matches

def main(args):
    if args.api_keys:
        for k in args.api_keys:
            if k.startswith('sk-'):
                os.environ["OPENAI_API_KEY"] = k
            elif k.startswith('esecret'):
                os.environ["ANYSCALE_API_KEY"] = k
            else:
                os.environ["DEEPINFRA_API_KEY"] = k

    utils.set_seed(args.seed)

    for game_name in args.game_names:
        run_game(game_name)


if __name__ == '__main__':
    args = get_args()
    for g in args.game_names:
        run_game(g)
