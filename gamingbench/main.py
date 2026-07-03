import copy
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
         'prisoners_dilemma']


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
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of games per LTM update batch. Default 1 = update after every game (existing behaviour).')
    args = parser.parse_args()

    return args


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
        return snap
    elif hasattr(agent, 'rules') and hasattr(agent, 'experience_pool'):
        return {
            'rules': copy.deepcopy(agent.rules),
            'experience_pool': copy.deepcopy(agent.experience_pool)
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
        clone.ltm_store_path = '/dev/null'
        clone.self_ltm_store_path = '/dev/null'
    if hasattr(clone, 'rules') and hasattr(clone, 'experience_pool'):
        if 'rules' in snapshot:
            clone.rules = copy.deepcopy(snapshot['rules'])
        if 'experience_pool' in snapshot:
            clone.experience_pool = copy.deepcopy(snapshot['experience_pool'])

def clone_agent_for_batch(original_agent, memory_snapshot: dict):
    """Create an independent agent copy seeded with a frozen memory snapshot.

    The clone:
    - Has its own in-memory store pre-loaded from memory_snapshot (no file I/O during game).
    - Has batch_mode=True so post_game_update() stores gradient data and returns early.
    - Does NOT call set_storage_dir() — it never writes to disk (store_path='/dev/null').
    """
    clone = copy.deepcopy(original_agent)
    clone._parent_store_path = getattr(original_agent, 'ew_store_path', getattr(original_agent, 'sw_store_path', getattr(original_agent, 'ltm_store_path', getattr(original_agent, 'rules_store_path', None))))
    _restore_memory_snapshot(clone, memory_snapshot)
    clone.batch_mode = True
    clone._last_batch_result = None
    return clone

def run_game(game_name):
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
    agents = [utils.load_agent(config_path, game=game.game)
              for config_path in args.agent_configs]
    models = [utils.load_model(config_path)
              for config_path in args.model_configs]

    for a, m in zip(agents, models):
        a.set_model(m)
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        if hasattr(a, 'set_storage_dir'):
            a.set_storage_dir(log_root)

    # exchange first player to mitigate first-player advantage
    reversed_agent_configs = copy.deepcopy(args.agent_configs)
    reversed_agent_configs.reverse()
    reversed_agents = [utils.load_agent(
        config_path, game=game.game) for config_path in reversed_agent_configs]
    reversed_model_configs = copy.deepcopy(args.model_configs)
    reversed_model_configs.reverse()
    reversed_models = [utils.load_model(config_path)
                       for config_path in reversed_model_configs]

    for a, m in zip(reversed_agents, reversed_models):
        a.set_model(m)
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        if hasattr(a, 'set_storage_dir'):
            a.set_storage_dir(log_root)

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
                store_path = getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', None))))
                if store_path and store_path not in seen_store_paths:
                    seen_store_paths.add(store_path)
                    batch_agents.append(a)
        results = []
        for batch_start in range(0, args.num_matches, batch_size):
            batch_end = min(batch_start + batch_size, args.num_matches)
            batch_q = queue.Queue()  # thread-safe gradient data collector

            # Freeze the current memory store state for all games in this batch
            memory_snapshots = {
                getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', None)))): _get_memory_snapshot(a) 
                for a in batch_agents
            }

            # Build per-game args, each with its own agent clones
            batch_args = []
            for match_idx in range(batch_start, batch_end):
                fresh_agents = []
                for a in agents:
                    if hasattr(a, 'flush_batch_updates'):
                        store_path = getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', None))))
                        snap = memory_snapshots.get(store_path, _get_memory_snapshot(a))
                        fresh_agents.append(clone_agent_for_batch(a, snap))
                    else:
                        fresh_agents.append(copy.deepcopy(a))

                fresh_reversed = []
                for a in reversed_agents:
                    if hasattr(a, 'flush_batch_updates'):
                        store_path = getattr(a, 'ew_store_path', getattr(a, 'sw_store_path', getattr(a, 'ltm_store_path', getattr(a, 'rules_store_path', None))))
                        snap = memory_snapshots.get(store_path, _get_memory_snapshot(a))
                        fresh_reversed.append(clone_agent_for_batch(a, snap))
                    else:
                        fresh_reversed.append(copy.deepcopy(a))
                for fa, m in zip(fresh_agents, models):
                    fa.set_model(m)
                    fa.enable_chat = getattr(args, 'enable_chat', False)
                    fa.think_further = getattr(args, 'think_further', False)
                for fa, m in zip(fresh_reversed, reversed_models):
                    fa.set_model(m)
                    fa.enable_chat = getattr(args, 'enable_chat', False)
                    fa.think_further = getattr(args, 'think_further', False)

                batch_args.append({
                    'match_idx': match_idx,
                    'game_name': game_name,
                    'agents': fresh_agents,
                    'reversed_agents': fresh_reversed,
                    'models': models,
                    'reversed_models': reversed_models,
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
                # Borrow opponent context (current_opponent_key / game_intro) from
                # the first successful clone that actually played.
                # Must check BOTH 'agents' and 'reversed_agents' because odd matches
                # (exchange_first_player) run with reversed_agents.
                for ba in batch_args:
                    sample = next(
                        (a for a in ba['agents'] + ba['reversed_agents']
                         if hasattr(a, 'current_opponent_key') and a.current_opponent_key),
                        None
                    )
                    if sample:
                        for batch_agent in batch_agents:
                            batch_agent.current_opponent_key = sample.current_opponent_key
                            batch_agent.current_game_intro  = sample.current_game_intro
                            batch_agent.current_game_name = game_name
                        break
                from gamingbench.prompts.observation_prompts import construct_game_intro
                for batch_agent in batch_agents:
                    batch_agent.current_game_name = game_name
                    if not getattr(batch_agent, 'current_game_intro', None):
                        batch_agent.current_game_intro = construct_game_intro(game_name)
                    store_path = getattr(batch_agent, 'ew_store_path', getattr(batch_agent, 'sw_store_path', getattr(batch_agent, 'ltm_store_path', getattr(batch_agent, 'rules_store_path', None))))
                    agent_data = [d for p_path, d in gradient_data if p_path == store_path]
                    if agent_data:
                        batch_agent.flush_batch_updates(agent_data)

        results = [r[0] for r in results]

    elif args.num_workers == 1:
        results = []
        for match_idx in range(args.num_matches):
            match_arg = {
                'match_idx': match_idx,
                'game_name': game_name,
                'agents': agents,
                'reversed_agents': reversed_agents,
                'models': models,
                'reversed_models': reversed_models,
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
                'reversed_models': reversed_models,
                'agents': agents,
                'reversed_agents': reversed_agents,
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
    reversed_agents = params['reversed_agents']
    models = params['models']
    reversed_models = params['reversed_models']
    result_path = params['result_path']

    args = params['args']
    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(
        args.game_config_root, f'{game_name}.yaml'))
    game_env.save_game_config(utils.load_config(
        os.path.join(args.game_config_root, f'{game_name}.yaml')))

    game_env.set_game(game)

    if args.exchange_first_player and match_idx % 2 == 1:
        # exchange first player
        game_env.set_agents(reversed_agents)
        game_env.set_models(reversed_models)
        reversed_agent_configs = copy.deepcopy(args.agent_configs)
        reversed_agent_configs.reverse()
        for config_path in reversed_agent_configs:
            game_env.append_agents_config(utils.load_config(config_path))

        reversed_model_configs = copy.deepcopy(args.model_configs)
        reversed_model_configs.reverse()
        for config_path in reversed_model_configs:
            game_env.append_models_config(utils.load_config(config_path))
    else:
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
        game_env.play()
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
        all_agents_in_match = list(params.get('agents', [])) + list(params.get('reversed_agents', []))
        for agent in all_agents_in_match:
            if getattr(agent, 'batch_mode', False) and agent._last_batch_result is not None:
                batch_queue.put((getattr(agent, '_parent_store_path', None), agent._last_batch_result))
                agent._last_batch_result = None

    return (res, params)



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
    main(args)
