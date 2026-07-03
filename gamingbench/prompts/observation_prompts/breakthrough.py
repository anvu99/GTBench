

def _construct_head_prompt():
    return """Breakthrough is a two-player game played on a rectangular board. Players take turns moving their pieces, which can move one space straight or diagonally forward if the target square is empty. A piece can only move diagonally forward to capture an opponent's piece (capturing straight forward is not allowed). Capturing is optional, and a player can only capture one piece per turn. The goal is to be the first to reach the opponent's home row, the farthest row from the player. If all of a player's pieces are captured, they lose. The game does not allow draws, as pieces can only move forward or be captured.The Breakthrough board is a 8x3 grid. It is identified by 3 columns labeled 'a', 'b', 'c' (from left to right) and 8 rows numbered 1 to 8 (from bottom to top). The intersection of a column and a row specifies a unique square on the board."""




def construct_observation_prompt(observations):

    legal_actions = observations.get('legal_moves', [])
    opponent_actions = observations.get('opponent_moves', [])
    agent_actions = observations.get('self_moves', [])
    board_str = observations.get('board', '')
    player_idx = observations.get('player_idx', 0)
    symbol = "Black ('b')" if player_idx == 0 else "White ('w')"

    if board_str == '':
        board_preview = f"You are playing as {symbol}."
    else:
        board_preview = (
            f"You are playing as {symbol}.\nThe board now looks like :\n{board_str}\n"
            f"Note: The numbers in the board string are the indexes of the rows. "
            f"The columns are 'a', 'b', 'c' from left to right (after the row number). "
            f"White ('w') pieces start at the bottom and MUST move upwards from row 1 towards row 8. "
            f"Black ('b') pieces start at the top and MUST move downwards from row 8 towards row 1.\n"
            f"The letter 'b' represents black piece, the letter 'w' represents white piece, and '.' represents vacant space."
        )

    if len(opponent_actions) == 0:
        opponent_prompt = 'Your opponent does not have any action so far.'
    else:
        finished_moves = ' and '.join(opponent_actions)
        opponent_prompt = f'Your opponent has finished actions: {finished_moves}.'

    if len(agent_actions) == 0:
        agent_prompt = 'You do not have any action so far.'
    else:
        finished_moves = ', '.join(agent_actions)
        agent_prompt = f'You have finished actions: {finished_moves}.'

    if len(legal_actions) == 0:
        legal_position_prompt = 'Currently, it is not your turn to act.'
    else:
        legal_pos = ' or '.join(legal_actions)
        legal_position_prompt = f'Currently, the legal actions are: {legal_pos}.'

    prompt = f'{board_preview}\n{opponent_prompt} {agent_prompt}\n\n{legal_position_prompt}'

    return prompt

def _construct_game_history_legend():
    return """- [Player Context]: States which piece color "You" and "Opponent" correspond to, and which direction they advance.
- [Position Legend]: Explains how to read the [Position] lines.
- [Position]: The board layout at the moment that player had to make their decision (before the move was executed). Rows listed top-to-bottom (Row 8 to Row 1); columns a-c left to right within each row. 'b'=Black, 'w'=White, '.'=empty."""
