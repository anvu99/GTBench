
def _construct_head_prompt():
    return 'You and your partner are in the Prisoner\'s Dilemma situation. ' \
           'Specifically, if you <Testify> against your partner and your partner remains <Silent>, you will go free while your partner will get 3 years in prison on the main charge. ' \
           'If you remain <Silent> but your partner <Testify> against you, you will serve 3 years in prison and your partner will be set free. ' \
           'If you and your partner <Testify> against each other, you and your partner will each serve 2 years. \n' \
           'If both you and your partner remain <Silent>, you and your partner will each server 1 year.'

def construct_observation_prompt(observations):

    opponent_moves = observations['opponent_moves']
    self_moves = observations['self_moves']
    if len(self_moves) > 0:
        past_round_prompt = 'You have been through some this situation in the past and here are the decisions you and your partner made:\n'
        for round_idx, (opp_char, self_char) in enumerate(zip(opponent_moves, self_moves)):
            self_move = '<Silent>' if self_char == 'C' else '<Testify>'
            opponent_move = '<Silent>' if opp_char == 'C' else '<Testify>'
            past_round_prompt += f'In the {round_idx+1}th round, you decided to {self_move} and your opponent decided to {opponent_move}.\n'
    else:
        past_round_prompt = ''

    prompt =              f'{past_round_prompt}' \
             f'\n' \
             'In this new round, you and your partner are making decision simultaneously and you do not know your partner\'s decision.' \
             '\n\n' \
             'The legal actions are: <Testify>, <Silent>.'

    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt