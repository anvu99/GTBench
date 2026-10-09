import copy
import logging
import os.path
import argparse
import pathlib
import queue
import random as _random_mod
import threading
from gamingbench.utils import utils
from gamingbench.environments.base_env import BaseGameEnv
import json

games = ['tictactoe', 'connect4', 'texas_holdem', 'texasholdem', 'neuron_poker', 'backgammon', 'breakthrough',
         'first_sealed_auction', 'gin_rummy', 'liars_dice', 'negotiation', 'nim', 'pig', 'kuhn_poker',
         'prisoners_dilemma', 'cooperative_negotiation', 'hanabi', 'hanabi-micro', 'hanabi-small', 'hanabi3-micro', 'hanabi-small-custom', 'mmlu_pro_qa',
         'buy_sell_game']


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
                        default=True, action='store_true')
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--threshold-matches', default=50, type=int)
    parser.add_argument('--enable-chat', default=False, action='store_true', help='Enable chat feature for agents')
    parser.add_argument('--think-further', default=False, action='store_true', help='Instruct agent to think multiple steps ahead')
    parser.add_argument('--n-player-memory-mode', type=str, default='combined', choices=['combined', 'separate'], help='Memory mode for N-player cooperative games')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of games per LTM update batch. Default 1 = update after every game (existing behaviour).')
    parser.add_argument('--use-strategy-memory', default=False, action='store_true',
                        help='Enable strategy memory for ProactiveQueryAgent. When absent, only the proactive query store is used.')
    parser.add_argument('--use-proactive-memory', default=False, action='store_true',
                        help='Enable proactive memory for ProactiveQueryAgent.')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing experiment directory based on jsonl line count.')
    parser.add_argument('--in-game-obs', default=False, action='store_true',
                        help='Enable in-game observation tracking for SlidingWindowAgent.')
    parser.add_argument('--sw-anti-decay', default=False, action='store_true',
                        help='Enable anti-decay mode for SlidingWindowAgent.')
    parser.add_argument('--sw-anti-contradiction', default=False, action='store_true',
                        help='Enable anti-contradiction mode for SlidingWindowAgent.')
    parser.add_argument('--sw-memory-mode', type=str, default='',
                        help='Memory mode for SlidingWindowAgent: reputation | strategy | dual | dual-structured | structured. '
                             'structured = dual-call (reputation then strategy); strategy brief uses DEFAULT/CONDITIONAL/AVOID format. '
                             'Empty string = default (bullet-list reputation).')
    parser.add_argument('--sw-track-frequency', default=False, action='store_true',
                        help='Enable per-observation frequency counting in SlidingWindowAgent memory updates.')
    parser.add_argument('--sw-trajectory-select', type=str, default='',
                        choices=['', 'best', 'worst', 'random', 'best_worst'],
                        help='Trajectory selection mode for SWA memory update: '
                             'best = top 8 by score, worst = bottom 8, random = random 8, '
                             'best_worst = top 4 + bottom 4 passed separately to synthesis prompt. '
                             'Requires --batch-size 16.')
    parser.add_argument('--sw-trajectories-included', type=int, default=8,
                        help='Number of trajectories to include when using '
                             '--sw-trajectory-select best/worst/random. Default: 8.')
    parser.add_argument('--sw-trajectory-balanced', default=False, action='store_true',
                        help='When used with --sw-trajectory-select, balance selected '
                             'trajectories across roles (e.g. 4 Seller + 4 Buyer in buy_sell_game).')
    parser.add_argument('--sw-explore-batch', type=int, default=0,
                        help='Size of the first (exploration) batch. When set, the first batch '
                             'uses this many games instead of --batch-size, giving the agent '
                             'more room to explore before its first memory update. '
                             'Subsequent batches use --batch-size as normal. 0 = disabled.')
    parser.add_argument('--st-use-confidence', default=False, action='store_true',
                        help='Enable confidence-scored strategy mode for SimpleTendencyAgent.')
    parser.add_argument('--st-use-explore', default=False, action='store_true',
                        help='Enable exploration mode with best-strategy version tracking for SimpleTendencyAgent.')
    parser.add_argument('--st-use-memory-only', default=False, action='store_true',
                        help='Memory-only mode for SimpleTendencyAgent: skip strategy synthesis '
                             'and inject raw reputation memory directly into the in-game prompt.')
    parser.add_argument('--llm-seed', type=int, default=None,
                        help='Seed for LLM sampling reproducibility. When set, vLLM seeds its RNG '
                             'per-request so the same prompt + seed yields the same output. '
                             'Overridden by llm_seed in the model YAML config if present.')
    parser.add_argument('--sw-explore-strategy', default=False, action='store_true',
                        help='Exploration mode for SlidingWindowAgent dual memory: tracks per-version '
                             'strategy utility, shows the best-performing strategy in the strategy '
                             'update prompt, and lets the LLM EVOLVE or [INHERIT BEST]. '
                             'Requires --sw-memory-mode dual.')
    parser.add_argument('--sw-deterministic-explore', default=False, action='store_true',
                        help='Deterministic exploration mode for SlidingWindowAgent: detects plateaus '
                             'using a patience window on batch averages and forces a 1-batch contrastive '
                             'exploration to break memory anchor.')
    args = parser.parse_args()

    if args.sw_explore_strategy:
        if args.sw_memory_mode not in ('dual', 'strategy'):
            parser.error('--sw-explore-strategy requires --sw-memory-mode dual or strategy')
        if args.sw_trajectory_select == 'best_worst':
            parser.error('--sw-explore-strategy is not supported with --sw-trajectory-select best_worst')

    return args


def _flush_agent_batch(batch_agent, agent_data, agent_items):
    """Call flush_batch_updates, passing per-game utilities for SW explore mode.

    Utilities come from ALL games in the batch (agent_items), not only the
    selected trajectories, because version tracking measures the strategy's
    true average performance.
    """
    if getattr(batch_agent, 'sw_explore_strategy', False) or getattr(batch_agent, 'sw_deterministic_explore', False):
        utilities = [item[2] for item in agent_items if len(item) >= 3]
        batch_agent.flush_batch_updates(agent_data, utilities=utilities)
    else:
        batch_agent.flush_batch_updates(agent_data)


def _get_agent_store_path(agent):
    """Helper to extract the correct store path for any memory agent."""
    return getattr(agent, 'ew_store_path', 
           getattr(agent, 'sw_store_path', 
           getattr(agent, 'ltm_store_path', 
           getattr(agent, 'rules_store_path', 
           getattr(agent, 'memory_bank_path', 
           getattr(agent, 'tendency_store_path',
           getattr(agent, 'store_path', None)))))))


# ── Game-state pregeneration ──────────────────────────────────────────────────
# Games with private information (negotiation, first_sealed_auction, liars_dice)
# need their random state generated *before* any parallel workers start so that:
#   1. The global RNG is advanced in a single deterministic sequence.
#   2. Paired exchange-first-player matches share the same private information.
#   3. Thread scheduling can never alter the outcomes.

_GAMES_WITH_CHANCE_NODES = {'first_sealed_auction', 'liars_dice', 'kuhn_poker', 'pig'}
_GAMES_WITH_CUSTOM_UTILS = {'negotiation'}
# All hanabi variant names as they appear in --game-name / YAML filenames
_HANABI_GAME_NAMES = {'hanabi', 'hanabi-micro', 'hanabi-small', 'hanabi3-micro', 'hanabi-small-custom'}


def _draw_game_state(game, game_name, rng):
    """Draw one match's random private information using *rng* (a random.Random instance).

    Returns a dict that run_match will apply directly to the game object.
    Supported keys:
      'custom_agent_utils'   – dict {player_idx: [v0, v1, v2]} for negotiation.
      'forced_chance_actions' – list of OpenSpiel chance actions for games with
                                 chance nodes (first_sealed_auction, liars_dice, …).
    """
    state = {}

    if game_name in _GAMES_WITH_CUSTOM_UTILS:
        # negotiation: two cut-points split 20 utility points across 3 items per player.
        custom_utils = {}
        for p in range(2):
            cuts = sorted(rng.sample(range(1, 20), 2))
            custom_utils[p] = [cuts[0], cuts[1] - cuts[0], 20 - cuts[1]]
        state['custom_agent_utils'] = custom_utils

    if game_name in _GAMES_WITH_CHANCE_NODES:
        # OpenSpiel games: inject our RNG into the game temporarily so that
        # _sample_chance_action uses it instead of global np.random.
        game._rng = rng
        game.reset()
        actions = []
        while game.env.is_chance_node():
            outcomes = game.env.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = game._sample_chance_action(action_list, prob_list)
            actions.append(action)
            game.env.apply_action(action)
        state['forced_chance_actions'] = actions

    if game_name in _HANABI_GAME_NAMES:
        # Generate a random integer seed for the C++ HanabiGame RNG.
        # Positive, 30-bit to fit safely in C++ int.
        state['hanabi_seed'] = rng.randint(0, 2 ** 30)

    return state


def _pregenerate_game_states(game, game_name, num_matches, seed, exchange_first_player):
    """Pre-generate all random private information for every match before workers start.

    A single isolated random.Random(seed) is advanced sequentially, so results
    are identical across runs regardless of thread scheduling or model __init__
    RNG consumption.

    For exchange_first_player: odd matches copy the preceding even match's state
    so paired games share identical private information while agents swap seats,
    giving each agent equal exposure to both valuation positions.

    Returns a list of game-state dicts, one per match (0..num_matches-1).
    """
    rng = _random_mod.Random(seed)
    states = []
    for match_idx in range(num_matches):
        if exchange_first_player and match_idx % 2 == 1:
            # Paired match: same private info, agents will swap seats in run_match.
            # Deep-copy so each match owns independent list objects — no shared
            # mutable state between parallel workers even for paired games.
            states.append(copy.deepcopy(states[-1]))
        else:
            states.append(_draw_game_state(game, game_name, rng))
    # Remove the temporary _rng we injected during chance-node sampling.
    if hasattr(game, '_rng'):
        del game._rng
    return states

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
    elif hasattr(agent, 'tendency_store'):
        return {
            'tendency_store': copy.deepcopy(agent.tendency_store)
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
    if hasattr(clone, 'tendency_store'):
        if 'tendency_store' in snapshot:
            clone.tendency_store = copy.deepcopy(snapshot['tendency_store'])
        clone.tendency_store_path = '/dev/null'

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
    if getattr(game_config, 'num_players', 2) != 2 or len(args.agent_configs) > 2:
        return run_game_nplayer(game_name)

    log_root = args.exp_root
    pathlib.Path(log_root).mkdir(parents=True, exist_ok=True)
    agent_names = [a.split('/')[-1].split('.')[0] for a in args.agent_configs]
    model_names = [m.split('/')[-1].split('.')[0] for m in args.model_configs]

    run_name = f'{agent_names[0]}_{model_names[0]}_{agent_names[1]}_{model_names[1]}'

    log_path = os.path.join(log_root, run_name + '.log')
    logger = utils.LLMBenchLogger(log_path)
    result_path = os.path.join(log_root, run_name + '.jsonl')

    start_match_idx = 0
    if getattr(args, 'resume', False) and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            start_match_idx = sum(1 for _ in f)
    elif not os.path.exists(result_path):
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
        m.nick_name = f"Player {i+1}"
        # Inject CLI --llm-seed into model if not already set via YAML config
        if getattr(args, 'llm_seed', None) is not None and getattr(m, 'llm_seed', None) is None:
            m.llm_seed = args.llm_seed
        a.set_model(m)
        a.player_id = f"p{i}"
        a.current_game_name = game_name.lower()
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        a.memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
        if hasattr(a, 'use_strategy_memory'):
            a.use_strategy_memory = getattr(args, 'use_strategy_memory', False)
        if hasattr(a, 'use_proactive_memory'):
            a.use_proactive_memory = getattr(args, 'use_proactive_memory', False)
        if hasattr(a, 'in_game_obs_mode'):
            a.in_game_obs_mode = getattr(args, 'in_game_obs', False)
        if hasattr(a, 'sw_anti_decay'):
            a.sw_anti_decay = getattr(args, 'sw_anti_decay', False)
        if hasattr(a, 'sw_anti_contradiction'):
            a.sw_anti_contradiction = getattr(args, 'sw_anti_contradiction', False)
        if hasattr(a, 'sw_memory_mode') and getattr(args, 'sw_memory_mode', ''):
            a.sw_memory_mode = args.sw_memory_mode
        if hasattr(a, 'sw_track_frequency'):
            a.sw_track_frequency = getattr(args, 'sw_track_frequency', False)
        if hasattr(a, 'sw_trajectory_select'):
            a.sw_trajectory_select = getattr(args, 'sw_trajectory_select', '')
        if hasattr(a, 'sw_trajectory_balanced'):
            a.sw_trajectory_balanced = getattr(args, 'sw_trajectory_balanced', False)
        if hasattr(a, 'sw_explore_strategy'):
            a.sw_explore_strategy = getattr(args, 'sw_explore_strategy', False)
        if hasattr(a, 'sw_deterministic_explore'):
            a.sw_deterministic_explore = getattr(args, 'sw_deterministic_explore', False)
        if hasattr(a, 'st_use_confidence'):
            a.st_use_confidence = getattr(args, 'st_use_confidence', False)
        if hasattr(a, 'st_use_explore'):
            a.st_use_explore = getattr(args, 'st_use_explore', False)
        if hasattr(a, 'st_use_memory_only'):
            a.st_use_memory_only = getattr(args, 'st_use_memory_only', False)
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
                key_a0 = f"{a0.agent_name}_{models[0].nick_name}"
                key_a1 = f"{a1.agent_name}_{models[1].nick_name}"
            a0.set_partner_store(a1.ltm_store_path, key_a0)
            a1.set_partner_store(a0.ltm_store_path, key_a1)



    for config_path in args.model_configs:
        game_env.append_models_config(utils.load_config(config_path))

    game_env.set_game(game)

    # Pre-generate all random game state (private valuations, chance actions) now,
    # in a single thread, before any parallel workers start. This guarantees that
    # the seed controls game setup deterministically regardless of thread scheduling.
    pregenerated_game_states = _pregenerate_game_states(
        game, game_name, args.num_matches, args.seed,
        exchange_first_player=getattr(args, 'exchange_first_player', False),
    )

    lock = threading.Lock()
    batch_size = getattr(args, 'batch_size', 1)
    explore_batch_size = getattr(args, 'sw_explore_batch', 0)

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
        for batch_start in _batch_starts(start_match_idx, args.num_matches, batch_size, explore_batch_size):
            # Use explore_batch_size for the first batch, normal batch_size after
            current_batch_size = explore_batch_size if (explore_batch_size > 0 and batch_start == start_match_idx) else batch_size
            batch_end = min(batch_start + current_batch_size, args.num_matches)
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

                if getattr(args, 'exchange_first_player', False) and match_idx % 2 == 1:
                    match_agents = [fresh_agents[1], fresh_agents[0]]
                    match_models = [models[1], models[0]]
                else:
                    match_agents = fresh_agents
                    match_models = list(models)

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
                    'game_state': pregenerated_game_states[match_idx],
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
                sw_traj_select = getattr(args, 'sw_trajectory_select', '')
                for batch_agent in batch_agents:
                    batch_agent.current_game_name = game_name.lower()
                    if not getattr(batch_agent, 'current_game_intro', None):
                        batch_agent.current_game_intro = construct_game_intro(game_name, game_config=game.config)
                    store_path = _get_agent_store_path(batch_agent)
                    
                    if store_path in flushed_paths:
                        continue
                    
                    # gradient_data items are (path, history, score) tuples
                    # when trajectory selection is active, otherwise (path, history) for backward compat
                    agent_items = [item for item in gradient_data if item[0] == store_path]
                    
                    if agent_items and sw_traj_select and len(agent_items[0]) >= 3:
                        # Trajectory selection: pick from the batch based on score
                        scored = [(item[1], item[2]) for item in agent_items]  # (history, score)
                        
                        log_file = store_path.replace('.json', '_trace.log') if store_path else None
                        sw_traj_balanced = getattr(args, 'sw_trajectory_balanced', False)
                        
                        if sw_traj_select == 'best_worst':
                            game_name_lower = game_name.lower()
                            if 'buy_sell' in game_name_lower:
                                threshold = 10.01
                            elif 'negotiation' in game_name_lower:
                                threshold = 50.0
                            elif 'first_sealed' in game_name_lower:
                                threshold = 0.01
                            else:
                                all_scores = [s for _, s in scored]
                                threshold = sum(all_scores) / len(all_scores) if all_scores else 0.0

                            pool_winning = [(h, s) for h, s in scored if s >= threshold]
                            pool_losing = [(h, s) for h, s in scored if s < threshold]
                            
                            sw_traj_incl = getattr(args, 'sw_trajectories_included', 8)
                            target_win, target_lose = sw_traj_incl // 2, sw_traj_incl - (sw_traj_incl // 2)
                            quota_win = min(target_win, len(pool_winning))
                            quota_lose = min(target_lose, len(pool_losing))
                            
                            if quota_lose < target_lose:
                                quota_win = min(len(pool_winning), quota_win + (target_lose - quota_lose))
                            elif quota_win < target_win:
                                quota_lose = min(len(pool_losing), quota_lose + (target_win - quota_win))
                                
                            if sw_traj_balanced:
                                winning = _get_top_k_balanced(pool_winning, quota_win, reverse=True)
                                losing = _get_top_k_balanced(pool_losing, quota_lose, reverse=False)
                            else:
                                pool_winning.sort(key=lambda x: x[1], reverse=True)
                                pool_losing.sort(key=lambda x: x[1], reverse=False)
                                winning = pool_winning[:quota_win]
                                losing = pool_losing[:quota_lose]
                                
                            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                            excluded = [item for item in scored_sorted if item not in winning and item not in losing]
                            
                            if log_file:
                                with open(log_file, 'a') as f:
                                    f.write(f'=== TRAJECTORY SELECTION (mode=best_worst, batch_start={batch_start}) ===\n')
                                    f.write(f'Total games: {len(scored)}, Hardcoded Threshold: {threshold:.2f}\n')
                                    f.write(f'Winning (Quota={quota_win}): {len(winning)}, Losing (Quota={quota_lose}): {len(losing)}\n')
                                    f.write(f'Winning scores: {[s for _, s in winning]}\n')
                                    f.write(f'Losing scores: {[s for _, s in losing]}\n')
                                    f.write(f'Excluded scores: {[s for _, s in excluded]}\n')
                                    f.write('=' * 50 + '\n\n')
                            
                            flushed_paths.add(store_path)
                            batch_agent.flush_batch_updates_best_worst(winning, losing)
                        else:
                            select_n = min(getattr(args, 'sw_trajectories_included', 8), len(scored))
                            if sw_traj_balanced:
                                selected, excluded = _balanced_trajectory_select(
                                    scored, select_n, sw_traj_select, batch_start, args.seed)
                            else:
                                selected, excluded = _flat_trajectory_select(
                                    scored, select_n, sw_traj_select, batch_start, args.seed)
                            
                            agent_data = [h for h, s in selected]
                            
                            # Log selection details
                            if log_file:
                                with open(log_file, 'a') as f:
                                    balanced_tag = ' [BALANCED]' if sw_traj_balanced else ''
                                    f.write(f'=== TRAJECTORY SELECTION (mode={sw_traj_select}{balanced_tag}, batch_start={batch_start}) ===\n')
                                    f.write(f'Total games: {len(scored)}, Selected: {len(selected)}\n')
                                    if sw_traj_balanced:
                                        sel_roles = ['Seller' if 'Role=Seller' in h else 'Buyer' if 'Role=Buyer' in h else '?' for h, _ in selected]
                                        f.write(f'Selected roles: {sel_roles}\n')
                                    f.write(f'Selected scores: {[s for _, s in selected]}\n')
                                    f.write(f'Excluded scores: {[s for _, s in excluded]}\n')
                                    f.write('=' * 50 + '\n\n')
                            
                            if agent_data:
                                flushed_paths.add(store_path)
                                _flush_agent_batch(batch_agent, agent_data, agent_items)
                    else:
                        # No trajectory selection: use all histories (backward compatible)
                        agent_data = [item[1] for item in agent_items]
                    
                        if agent_data:
                            flushed_paths.add(store_path)
                            _flush_agent_batch(batch_agent, agent_data, agent_items)

        results = [r[0] for r in results]

    elif args.num_workers == 1:
        results = []
        for match_idx in range(start_match_idx, args.num_matches):
            if getattr(args, 'exchange_first_player', False) and match_idx % 2 == 1:
                match_agents = [copy.deepcopy(agents[1]), copy.deepcopy(agents[0])]
                match_models = [models[1], models[0]]
            else:
                match_agents = [copy.deepcopy(a) for a in agents]
                match_models = list(models)
                
            match_arg = {
                'match_idx': match_idx,
                'game_name': game_name,
                'agents': match_agents,
                'models': match_models,
                'result_path': result_path,
                'args': args,
                'lock': lock,
                'game_state': pregenerated_game_states[match_idx],
            }
            results.append(run_match(match_arg))
    else:
        match_arg_list = []
        for match_idx in range(start_match_idx, args.num_matches):
            if getattr(args, 'exchange_first_player', False) and match_idx % 2 == 1:
                match_agents = [copy.deepcopy(agents[1]), copy.deepcopy(agents[0])]
                match_models = [models[1], models[0]]
            else:
                match_agents = [copy.deepcopy(a) for a in agents]
                match_models = list(models)

            match_arg_list.append({
                'match_idx': match_idx,
                'game_name': game_name,
                'models': match_models,
                'agents': match_agents,
                'result_path': result_path,
                'args': args,
                'lock': lock,
                'game_state': pregenerated_game_states[match_idx],
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


def _batch_starts(start_idx, num_matches, batch_size, explore_batch_size=0):
    """Generate batch start indices, optionally using a larger first batch.
    
    If explore_batch_size > 0, the first batch spans [start_idx, start_idx + explore_batch_size),
    then subsequent batches use the normal batch_size.
    If explore_batch_size == 0, behaves identically to range(start_idx, num_matches, batch_size).
    
    Args:
        start_idx: starting match index (usually 0, or resuming from a checkpoint)
        num_matches: total number of matches to play
        batch_size: normal batch size for batches 2, 3, ...
        explore_batch_size: size of the first (exploration) batch; 0 = disabled
    
    Yields:
        batch_start indices
    """
    pos = start_idx
    first = True
    while pos < num_matches:
        yield pos
        if first and explore_batch_size > 0:
            pos += explore_batch_size
            first = False
        else:
            pos += batch_size


def _flat_trajectory_select(scored, select_n, mode, batch_start, seed):
    """Select trajectories using a single flat list (original behavior).
    
    Args:
        scored: list of (history, score) tuples
        select_n: number to select
        mode: 'best' | 'worst' | 'random'
        batch_start: batch start index (for random seed)
        seed: base seed
    
    Returns:
        (selected, excluded) tuple of lists of (history, score)
    """
    if mode == 'best':
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        return scored_sorted[:select_n], scored_sorted[select_n:]
    elif mode == 'worst':
        scored_sorted = sorted(scored, key=lambda x: x[1])
        return scored_sorted[:select_n], scored_sorted[select_n:]
    elif mode == 'random':
        import random as _rnd
        rng = _rnd.Random(batch_start + seed)
        indices = list(range(len(scored)))
        rng.shuffle(indices)
        selected = [scored[i] for i in indices[:select_n]]
        excluded = [scored[i] for i in indices[select_n:]]
        return selected, excluded
    else:
        return list(scored), []

def _get_top_k_balanced(pool, quota, reverse=True):
    if not pool:
        return []
    sellers = sorted([x for x in pool if 'Role=Seller' in x[0]], key=lambda x: x[1], reverse=reverse)
    buyers = sorted([x for x in pool if 'Role=Buyer' in x[0]], key=lambda x: x[1], reverse=reverse)
    unknown = sorted([x for x in pool if 'Role=Seller' not in x[0] and 'Role=Buyer' not in x[0]], key=lambda x: x[1], reverse=reverse)
    
    half = quota // 2
    take_s = min(half, len(sellers))
    take_b = min(half, len(buyers))
    
    deficit_s = half - take_s
    deficit_b = half - take_b
    
    take_s = min(len(sellers), take_s + deficit_b)
    take_b = min(len(buyers), take_b + deficit_s)
    
    selected = sellers[:take_s] + buyers[:take_b]
    
    remaining = quota - len(selected)
    if remaining > 0:
        leftovers = sellers[take_s:] + buyers[take_b:] + unknown
        leftovers.sort(key=lambda x: x[1], reverse=reverse)
        selected.extend(leftovers[:remaining])
        
    return selected


def _balanced_trajectory_select(scored, select_n, mode, batch_start, seed):
    """Select trajectories while balancing across roles.
    
    Partitions games by role (detecting 'Role=Seller'/'Role=Buyer' in the
    history string for buy_sell_game), then applies the selection mode within
    each group independently to guarantee equal representation.
    
    Falls back to flat selection if role markers are not found.
    
    Args:
        scored: list of (history, score) tuples
        select_n: total number to select (will be split evenly across groups)
        mode: 'best' | 'worst' | 'random'
        batch_start: batch start index (for random seed)
        seed: base seed
    
    Returns:
        (selected, excluded) tuple of lists of (history, score)
    """
    # Partition by role
    group_seller = []
    group_buyer = []
    group_unknown = []
    
    for h, s in scored:
        if 'Role=Seller' in h:
            group_seller.append((h, s))
        elif 'Role=Buyer' in h:
            group_buyer.append((h, s))
        else:
            group_unknown.append((h, s))
    
    # If we can't detect roles, fall back to flat selection
    if not group_seller or not group_buyer:
        return _flat_trajectory_select(scored, select_n, mode, batch_start, seed)
    
    # Split select_n evenly across groups
    per_group = select_n // 2
    
    # Select from each group independently
    sel_a, exc_a = _flat_trajectory_select(group_seller, min(per_group, len(group_seller)), mode, batch_start, seed)
    sel_b, exc_b = _flat_trajectory_select(group_buyer, min(per_group, len(group_buyer)), mode, batch_start, seed + 1)
    
    # If one group is short, take extra from the other
    total_selected = len(sel_a) + len(sel_b)
    if total_selected < select_n:
        deficit = select_n - total_selected
        # Try to fill from the group that has more excluded items
        if len(exc_a) >= deficit:
            sel_a.extend(exc_a[:deficit])
            exc_a = exc_a[deficit:]
        elif len(exc_b) >= deficit:
            sel_b.extend(exc_b[:deficit])
            exc_b = exc_b[deficit:]
    
    # Unknown-role games go to excluded
    selected = sel_a + sel_b
    excluded = exc_a + exc_b + group_unknown
    
    return selected, excluded


def run_match(params):
    match_idx = params['match_idx']
    game_name = params['game_name']
    agents = params['agents']
    models = params['models']
    result_path = params['result_path']

    args = params['args']
    game_state = params.get('game_state', {})

    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(
        args.game_config_root, f'{game_name}.yaml'))

    # Apply the pregenerated private information for this match.
    # custom_agent_utils: negotiation private valuations, set directly (no RNG needed).
    if 'custom_agent_utils' in game_state:
        game.custom_agent_utils = {int(k): list(v) for k, v in game_state['custom_agent_utils'].items()}
    elif hasattr(game, '_init_custom_utils') and not game.custom_agent_utils:
        # Fallback for games not covered by pregeneration.
        game._rng = _random_mod.Random(args.seed + match_idx)
        game._init_custom_utils()

    # forced_chance_actions: OpenSpiel private deals (valuations, card hands, dice).
    game.forced_chance_actions = list(game_state.get('forced_chance_actions', []))

    # Set a per-match isolated RNG for any runtime chance nodes
    # (tie-breakers, mid-game dice rolls) that appear after
    # forced_chance_actions are exhausted.
    game._rng = _random_mod.Random(args.seed + match_idx)

    # hanabi_seed: deterministic card deal for the hanabi_learning_environment backend.
    if 'hanabi_seed' in game_state:
        game._forced_seed = game_state['hanabi_seed']

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
        first_player = 0
        game_env.play(first_player=first_player)
        res = game_env.history_tracker.to_dict()
    finally:
        utils.remove_game_log_handler(game_log_handler)

    with params['lock']:
        with open(result_path, 'a') as file:
            file.writelines(json.dumps(res) + '\n')

    # ── Extract per-agent scores for trajectory selection ──────────────────────
    match_data = res.get('matches', [{}])[0]
    winner_name = match_data.get('winner', '')
    winner_score = match_data.get('winner_score', 0) or 0
    loser_score = match_data.get('loser_score', 0) or 0
    for i, agent in enumerate(agents):
        agent_name = getattr(agent, 'agent_name', f'Agent{i}')
        model_name = models[i].nick_name if i < len(models) else ''
        full_name = f"{agent_name}_{model_name}"
        if winner_name == "":
            agent._match_score = winner_score
        elif full_name in winner_name:
            agent._match_score = winner_score
        else:
            agent._match_score = loser_score

    # ── Batch mode: deposit gradient data into the shared queue ──────────────
    # Check BOTH agents and reversed_agents: for odd matches (exchange_first_player)
    # the game runs with reversed_agents, so the LTM clone in reversed_agents has
    # the _last_batch_result, not the one in agents.
    batch_queue = params.get('batch_queue')
    if batch_queue is not None:
        all_agents_in_match = list(params.get('agents', []))
        for agent in all_agents_in_match:
            if getattr(agent, 'batch_mode', False) and agent._last_batch_result is not None:
                score = getattr(agent, '_match_score', 0)
                batch_queue.put((getattr(agent, '_parent_store_path', None), agent._last_batch_result, score))
                agent._last_batch_result = None

    return (res, params)



def run_match_nplayer(params):
    match_idx = params['match_idx']
    game_name = params['game_name']
    result_path = params['result_path']
    lock = params['lock']
    args = params['args']
    log_root = params['log_root']
    forced_chance_actions = params.get('forced_chance_actions', [])

    game_env = BaseGameEnv()
    game = utils.load_game(os.path.join(args.game_config_root, f'{game_name}.yaml'))

    # Inject a per-game isolated RNG (same logic as run_match).
    import random as _random_mod
    game._rng = _random_mod.Random(args.seed + match_idx)
    if hasattr(game, '_init_custom_utils'):
        game._init_custom_utils()

    # inject the forced chance actions into the game
    game.forced_chance_actions = list(forced_chance_actions)

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
                        opponent_keys.append(f"{other_agent.agent_name}_{models[j].nick_name}")
            
            memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
            if memory_mode == 'combined':
                agent.reset_game_state("+".join(sorted(opponent_keys)), game_intro)
            else:
                agent.reset_game_state(sorted(opponent_keys), game_intro)

    try:
        from gamingbench.utils.history_tracker import HistoryTracker
        tracker = HistoryTracker()
        game_env.game.play(agents, models, tracker, seat_mapping=params.get('seat_mapping'), match_idx=match_idx)
        logger.info(f'Game {match_idx} ends')
        logger.info(tracker.matches[-1].to_dict())

        match_dict = tracker.matches[-1].to_dict()
        match_dict['match_idx'] = match_idx

        with lock:
            with open(result_path, 'a') as f:
                f.write(json.dumps(match_dict) + '\n')

    finally:
        utils.remove_game_log_handler(game_log_handler)
        
    # ── Extract per-agent scores for trajectory selection ──────────────────────
    winner_name = match_dict.get('winner', '')
    winner_score = match_dict.get('winner_score', 0) or 0
    loser_score = match_dict.get('loser_score', 0) or 0
    for i, agent in enumerate(agents):
        agent_name = getattr(agent, 'agent_name', f'Agent{i}')
        model_name = models[i].nick_name if i < len(models) else ''
        full_name = f"{agent_name}_{model_name}"
        if winner_name == "":
            agent._match_score = winner_score
        elif full_name in winner_name:
            agent._match_score = winner_score
        else:
            agent._match_score = loser_score

    # ── Batch mode: deposit gradient data into the shared queue ──────────────
    batch_queue = params.get('batch_queue')
    if batch_queue is not None:
        all_agents_in_match = list(params.get('agents', []))
        for agent in all_agents_in_match:
            if getattr(agent, 'batch_mode', False) and agent._last_batch_result is not None:
                score = getattr(agent, '_match_score', 0)
                batch_queue.put((getattr(agent, '_parent_store_path', None), agent._last_batch_result, score))
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

    start_match_idx = 0
    if getattr(args, 'resume', False) and os.path.exists(result_path):
        with open(result_path, 'r') as f:
            start_match_idx = sum(1 for _ in f)
    elif not os.path.exists(result_path):
        with open(result_path, 'w') as file:
            pass

    game = utils.load_game(os.path.join(args.game_config_root, f'{game_name}.yaml'))
    agents = [utils.load_agent(config_path, game=game.game, game_config=game.config) for config_path in agent_configs]
    models = [utils.load_model(config_path) for config_path in model_configs]

    for i, (a, m) in enumerate(zip(agents, models)):
        m.nick_name = f"Player {i+1}"
        # Inject CLI --llm-seed into model if not already set via YAML config
        if getattr(args, 'llm_seed', None) is not None and getattr(m, 'llm_seed', None) is None:
            m.llm_seed = args.llm_seed
        a.set_model(m)
        a.player_id = f"p{i}"
        a.current_game_name = game_name.lower()
        a.enable_chat = getattr(args, 'enable_chat', False)
        a.think_further = getattr(args, 'think_further', False)
        a.memory_mode = getattr(args, 'n_player_memory_mode', 'combined')
        if hasattr(a, 'use_strategy_memory'):
            a.use_strategy_memory = getattr(args, 'use_strategy_memory', False)
        if hasattr(a, 'use_proactive_memory'):
            a.use_proactive_memory = getattr(args, 'use_proactive_memory', False)
        if hasattr(a, 'in_game_obs_mode'):
            a.in_game_obs_mode = getattr(args, 'in_game_obs', False)
        if hasattr(a, 'sw_anti_decay'):
            a.sw_anti_decay = getattr(args, 'sw_anti_decay', False)
        if hasattr(a, 'sw_anti_contradiction'):
            a.sw_anti_contradiction = getattr(args, 'sw_anti_contradiction', False)
        if hasattr(a, 'sw_memory_mode') and getattr(args, 'sw_memory_mode', ''):
            a.sw_memory_mode = args.sw_memory_mode
        if hasattr(a, 'sw_track_frequency'):
            a.sw_track_frequency = getattr(args, 'sw_track_frequency', False)
        if hasattr(a, 'sw_trajectory_select'):
            a.sw_trajectory_select = getattr(args, 'sw_trajectory_select', '')
        if hasattr(a, 'sw_trajectory_balanced'):
            a.sw_trajectory_balanced = getattr(args, 'sw_trajectory_balanced', False)
        if hasattr(a, 'sw_explore_strategy'):
            a.sw_explore_strategy = getattr(args, 'sw_explore_strategy', False)
        if hasattr(a, 'st_use_confidence'):
            a.st_use_confidence = getattr(args, 'st_use_confidence', False)
        if hasattr(a, 'st_use_explore'):
            a.st_use_explore = getattr(args, 'st_use_explore', False)
        if hasattr(a, 'st_use_memory_only'):
            a.st_use_memory_only = getattr(args, 'st_use_memory_only', False)
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
                key_a0 = f"{a0.agent_name}_{models[0].nick_name}"
                key_a1 = f"{a1.agent_name}_{models[1].nick_name}"
            a0.set_partner_store(a1.ltm_store_path, key_a0)
            a1.set_partner_store(a0.ltm_store_path, key_a1)

    lock = threading.Lock()
    batch_size = getattr(args, 'batch_size', 1)
    explore_batch_size = getattr(args, 'sw_explore_batch', 0)
    
    if batch_size > 1:
        # ── Batch mode ────────────────────────────────────────────────────────
        seen_store_paths = set()
        batch_agents = []
        for a in agents:
            if hasattr(a, 'flush_batch_updates'):
                store_path = _get_agent_store_path(a)
                if store_path and store_path not in seen_store_paths:
                    seen_store_paths.add(store_path)
                    batch_agents.append(a)
        results = []
        for batch_start in _batch_starts(start_match_idx, args.num_matches, batch_size, explore_batch_size):
            current_batch_size = explore_batch_size if (explore_batch_size > 0 and batch_start == start_match_idx) else batch_size
            batch_end = min(batch_start + current_batch_size, args.num_matches)
            batch_q = queue.Queue()  # thread-safe gradient data collector

            memory_snapshots = {
                _get_agent_store_path(a): _get_memory_snapshot(a) 
                for a in batch_agents
            }

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
                sw_traj_select = getattr(args, 'sw_trajectory_select', '')
                for batch_agent in batch_agents:
                    batch_agent.current_game_name = game_name.lower()
                    if not getattr(batch_agent, 'current_game_intro', None):
                        # game config is not easily accessible here? We need game.config
                        batch_agent.current_game_intro = construct_game_intro(game_name, game_config=game.config)
                    store_path = _get_agent_store_path(batch_agent)
                    
                    if store_path in flushed_paths:
                        continue
                        
                    agent_items = [item for item in gradient_data if item[0] == store_path]
                    
                    if agent_items and sw_traj_select and len(agent_items[0]) >= 3:
                        scored = [(item[1], item[2]) for item in agent_items]
                        sw_traj_balanced = getattr(args, 'sw_trajectory_balanced', False)
                        log_file = store_path.replace('.json', '_trace.log') if store_path else None
                        
                        if sw_traj_select == 'best_worst':
                            game_name_lower = game_name.lower()
                            if 'buy_sell' in game_name_lower:
                                threshold = 10.01
                            elif 'negotiation' in game_name_lower:
                                threshold = 50.0
                            elif 'first_sealed' in game_name_lower:
                                threshold = 0.01
                            else:
                                all_scores = [s for _, s in scored]
                                threshold = sum(all_scores) / len(all_scores) if all_scores else 0.0

                            pool_winning = [(h, s) for h, s in scored if s >= threshold]
                            pool_losing = [(h, s) for h, s in scored if s < threshold]
                            
                            sw_traj_incl = getattr(args, 'sw_trajectories_included', 8)
                            target_win, target_lose = sw_traj_incl // 2, sw_traj_incl - (sw_traj_incl // 2)
                            quota_win = min(target_win, len(pool_winning))
                            quota_lose = min(target_lose, len(pool_losing))
                            
                            if quota_lose < target_lose:
                                quota_win = min(len(pool_winning), quota_win + (target_lose - quota_lose))
                            elif quota_win < target_win:
                                quota_lose = min(len(pool_losing), quota_lose + (target_win - quota_win))
                                
                            if sw_traj_balanced:
                                winning = _get_top_k_balanced(pool_winning, quota_win, reverse=True)
                                losing = _get_top_k_balanced(pool_losing, quota_lose, reverse=False)
                            else:
                                pool_winning.sort(key=lambda x: x[1], reverse=True)
                                pool_losing.sort(key=lambda x: x[1], reverse=False)
                                winning = pool_winning[:quota_win]
                                losing = pool_losing[:quota_lose]
                                
                            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                            excluded = [item for item in scored_sorted if item not in winning and item not in losing]
                            
                            if log_file:
                                with open(log_file, 'a') as f:
                                    f.write(f'=== TRAJECTORY SELECTION (mode=best_worst, batch_start={batch_start}) ===\n')
                                    f.write(f'Total games: {len(scored)}, Hardcoded Threshold: {threshold:.2f}\n')
                                    f.write(f'Winning (Quota={quota_win}): {len(winning)}, Losing (Quota={quota_lose}): {len(losing)}\n')
                                    f.write(f'Winning scores: {[s for _, s in winning]}\n')
                                    f.write(f'Losing scores: {[s for _, s in losing]}\n')
                                    f.write(f'Excluded scores: {[s for _, s in excluded]}\n')
                                    f.write('=' * 50 + '\n\n')
                            
                            flushed_paths.add(store_path)
                            batch_agent.flush_batch_updates_best_worst(winning, losing)
                        else:
                            select_n = min(getattr(args, 'sw_trajectories_included', 8), len(scored))
                            if sw_traj_balanced:
                                selected, excluded = _balanced_trajectory_select(
                                    scored, select_n, sw_traj_select, batch_start, args.seed)
                            else:
                                selected, excluded = _flat_trajectory_select(
                                    scored, select_n, sw_traj_select, batch_start, args.seed)
                            
                            agent_data = [h for h, s in selected]
                            
                            if log_file:
                                with open(log_file, 'a') as f:
                                    balanced_tag = ' [BALANCED]' if sw_traj_balanced else ''
                                    f.write(f"=== BATCH SELECTION (Mode: {sw_traj_select}{balanced_tag}) ===\n")
                                    f.write(f"Selected {len(selected)} out of {len(scored)} games.\n")
                                    if sw_traj_balanced:
                                        sel_roles = ['Seller' if 'Role=Seller' in h else 'Buyer' if 'Role=Buyer' in h else '?' for h, _ in selected]
                                        f.write(f"Selected roles: {sel_roles}\n")
                                    f.write("Selected scores: " + ", ".join(str(s) for h, s in selected) + "\n")
                                    f.write("Excluded scores: " + ", ".join(str(s) for h, s in excluded) + "\n")
                                    f.write("="*50 + "\n\n")
                            
                            if agent_data:
                                flushed_paths.add(store_path)
                                _flush_agent_batch(batch_agent, agent_data, agent_items)
                    else:
                        agent_data = [item[1] for item in agent_items]
                        if agent_data:
                            flushed_paths.add(store_path)
                            _flush_agent_batch(batch_agent, agent_data, agent_items)

    elif args.num_workers == 1:
        for match_idx in range(start_match_idx, args.num_matches):
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
        for match_idx in range(start_match_idx, args.num_matches):
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

    utils.set_seed(args.seed)  # Initial seed for agent/model init reproducibility.

    for game_name in args.game_names:
        run_game(game_name)


if __name__ == '__main__':
    args = get_args()
    main(args)
