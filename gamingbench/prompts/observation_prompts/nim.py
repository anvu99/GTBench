
def _construct_head_prompt():
    # avoid to take the last one.
    return 'In Nim, a strategic game with a set of four piles containing 1, 3, 5, and 7 matches respectively, ' \
           'players aim to avoid taking the last match. During each turn, a player may take any number of matches from a single pile, ' \
           'but must take at least one and cannot exceed the number remaining in that pile. ' \
           'The objective is to force the opponent to pick up the final match, thereby winning the game.' \
           '\n' \
           'The action is presented in <pile:x, take:y>, which means take y match(es) from the x-th pile.'


def construct_observation_prompt(observations):

    piles = observations['piles']

    legal_moves = observations.get('legal_moves', [])

    legal_move_str = ', '.join(legal_moves)

    prompt = f'Currently, the 1st pile has {piles[0]} match(es);\nthe 2nd pile has {piles[1]} match(es);\n' \
             f'the 3rd pile has {piles[2]} match(es);\nthe 4th pile has {piles[3]} match(es). \n\n' \
             f'The legal actions are: {legal_move_str}.'

    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt

def _construct_game_history_legend():
    return """- [Player Context]: States which player is "You" and "Opponent".
- [Position Legend]: Explains how to read the [Position] lines.
- [Position]: The state of the game at the moment that player had to make their decision (before the move was executed). It shows the number of matches remaining in each of the 4 piles."""


if __name__ == '__main__':
    observations = {'piles': [1, 3, 5, 7]}
    prompt = construct_observation_prompt(observations)
    print(prompt)
