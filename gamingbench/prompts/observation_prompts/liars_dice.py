
def _construct_head_prompt():
    # 1-4: at least 1 dice with face value as 4
    # the player may bid a higher quantity of any particular face, or the same quantity of a higher face (allowing a player to "re-assert" a face value they believe prevalent if another player increased the face value on their bid);
    return 'Liar\'s Dice is a game of bluffing and probability, played with two players and each player has 1 dice.' \
           'During each turn, a player can either bid a higher quantity of any particular face value or ' \
           'the same quantity of a higher face value than the previous bid. ' \
           'Each player tries to outbid their opponent without being caught in a lie. ' \
           '\n' \
           'The move in this game is denoted in <x dices, y value>, meaning there are at least x dices with face values as y.' \


def construct_observation_prompt(observations):

    self_dice_face_value = observations['self_dice_face_value']
    last_move = observations['last_move']
    legal_moves = observations['legal_moves']
    legal_move_str = ', '.join(legal_moves)

    if last_move is None:
        prompt = f'Currently, the face value of your dice is {self_dice_face_value}. You are the first to go.' \
                 '\n' \
                 'You are playing the Liar\'s Dice with another opponent. Therefore, there are only two dices in total.' \
                 f'\n\n' \
                 f'The legal actions are: {legal_move_str}.'
                 # 'You should call action <Liar> if the opponent called <2 dices, 6 value> in the last round. Because there is no other actions.' \
    else:
        prompt = f'Currently, the face value of your dice is {self_dice_face_value}. Last time, the opponent called action <{last_move}>.' \
                 '\n' \
                 'You are playing the Liar\'s Dice with another opponent. Therefore, there are only two dices in total.' \
                 f'\n\n' \
                 f'The legal actions are: {legal_move_str}.'
                 # 'You should call action <Liar> if the opponent called <2 dices, 6 value> in the last round. Because there is no other actions.' \

    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt


def _construct_game_history_legend():
    return """\
- [Your Dice]: The face value of your own die this turn (1–6). This is the only die value you can see — the opponent's die is hidden.
- [Opponent Dice]: Listed in the history as the opponent's die value, but this is ONLY revealed at the end of the game when a <Liar> challenge is made. During the game, each player only knows their own die.
- [Move]: The action taken this turn. Bids are written as <x dices, y value>, meaning the bidder claims there are at least x dice showing face value y across BOTH dice combined (2 total dice in the game). <Liar> means the player is challenging the previous bid — both dice are then revealed and the claim is verified.
- [Outcome]: At game end, if <Liar> was called, the true dice values are revealed. The challenger wins if the previous bid was false (actual count < claimed count); the bidder wins if the bid was true (actual count >= claimed count).\
"""