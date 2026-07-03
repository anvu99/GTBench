

def _construct_head_prompt():
    return """Tic Tac Toe is a two-player game played on a grid. Players take turns marking a space with their respective symbols. The goal is to get 3 of one\'s own symbols in a row, either horizontally, vertically, or diagonally, before the opponent does. If all nine squares are filled and no player has three in a row, the game is a draw. The Tic Tac Toe game is played on a 3 by 3 grid, with the winning length as 3.
Each cell is identified by a column (C1 to C3, from left to right) and a row (R1 to R3, from top to bottom). Row 1 is the top row, and Row 3 is the bottom row.
Each move is represented by a string consisting of two parts: the column (C) and the row (R), in that order. For instance, <C1R2> means placing a mark at the first column and the second row of the grid. You are playing this game with the user (opponent)."""

def construct_observation_prompt(observations):
    """
    :param observations: tic tac toe observation
    :return: observation prompts
    """

    legal_moves = observations.get('legal_moves', [])
    opponent_moves = observations.get('opponent_moves', [])
    self_moves = observations.get('self_moves', [])

    player_idx = observations.get('player_idx', 0)
    symbol = "X (Crosses)" if player_idx == 0 else "O (Noughts)"
    
    board_str = observations.get('board', '')
    if board_str:
        board_preview = f"The board currently looks like this:\n{board_str}\n"
    else:
        board_preview = ""

    if len(opponent_moves) != 0 or len(self_moves) != 0:
        if len(opponent_moves) == 0:
            opponent_prompt = ''
        else:
            finished_moves = ', '.join(opponent_moves)
            opponent_prompt = f'Your opponent has finished actions: {finished_moves}.'
        if len(self_moves) == 0:
            agent_prompt = ''
        else:
            finished_moves = ', '.join(self_moves)
            agent_prompt = f'You have finished actions: {finished_moves}.'
        finished_move_prompt = f'You are playing as {symbol}.\n{board_preview}\n{opponent_prompt} {agent_prompt}'
    else:
        finished_move_prompt = f'You are playing as {symbol}.\n{board_preview}\nYou are the first to go.'

    if len(legal_moves) == 0:
        legal_position_prompt = 'Currently, it is not your turn to act.'
    else:
        legal_pos = ', '.join(legal_moves)
        legal_position_prompt = f'Currently, the legal actions are {legal_pos}.'

    prompt = f'{finished_move_prompt}\n{legal_position_prompt}'

    return prompt

def _construct_game_history_legend():
    return """- [Player Context]: States which symbol ("x" or "o") "You" and "Opponent" correspond to.
- [Position Legend]: Explains how to read the [Position] lines.
- [Position]: The board layout at the moment that player had to make their decision (before the move was executed). The board is a 3x3 grid with cells labeled from C1R1 to C3R3, where rows are 1 (top) to 3 (bottom) and columns are 1 (left) to 3 (right). 'x'=Cross, 'o'=Nought, '.'=empty."""
