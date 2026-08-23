
from gamingbench.prompts.observation_prompts import connect4
from gamingbench.prompts.observation_prompts import tictactoe
from gamingbench.prompts.observation_prompts import breakthrough
from gamingbench.prompts.observation_prompts import first_sealed_auction
from gamingbench.prompts.observation_prompts import liars_dice
from gamingbench.prompts.observation_prompts import negotiation
from gamingbench.prompts.observation_prompts import nim
from gamingbench.prompts.observation_prompts import pig
from gamingbench.prompts.observation_prompts import kuhn_poker
from gamingbench.prompts.observation_prompts import prisoners_dilemma
from gamingbench.prompts.observation_prompts import cooperative_negotiation
from gamingbench.prompts.observation_prompts import hanabi
from gamingbench.prompts.observation_prompts import texas_holdem

# maps
mapping = {
    'connect4': connect4,
    'tictactoe': tictactoe,
    'breakthrough': breakthrough,
    'first_sealed_auction': first_sealed_auction,
    'liars_dice': liars_dice,
    'negotiation': negotiation,
    'nim': nim,
    'pig': pig,
    'kuhn_poker': kuhn_poker,
    'python_iterated_prisoners_dilemma': prisoners_dilemma,
    'prisoners_dilemma': prisoners_dilemma,
    'cooperative_negotiation': cooperative_negotiation,
    'hanabi': hanabi,
    'hanabi-micro': hanabi,
    'hanabi-small': hanabi,
    'hanabi-small-custom': hanabi,
    'hanabi3-micro': hanabi,
    'texas_holdem': texas_holdem
}

def construct_observation_prompt(observations, environment_name):

    return mapping[environment_name].construct_observation_prompt(observations)

def construct_game_intro(environment_name, enable_chat=False, game_config=None):
    if environment_name in ['cooperative_negotiation']:
        return mapping[environment_name]._construct_head_prompt(enable_chat=enable_chat)
    if environment_name.startswith('hanabi'):
        return mapping[environment_name]._construct_head_prompt(enable_chat=enable_chat, game_config=game_config)
    return mapping[environment_name]._construct_head_prompt()

def construct_game_history_legend(environment_name):
    return mapping[environment_name]._construct_game_history_legend()
