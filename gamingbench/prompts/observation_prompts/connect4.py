

def _construct_head_prompt():
    return """Connect 4 is a two-player connection board game, played on a 6x7 vertical grid. Players take turns dropping colored discs into one of the 7 columns. The pieces fall straight down, occupying the lowest available row index in that column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. You are a gaming agent that aims to beat the opponent in Connect 4 games.
    The columns are labeled C1 to C7 from left to right. The rows are numbered R1 (bottom) to R6 (top).
    You only choose the column to drop your piece into. Each move must be formatted as <Cx> where x is the column number (1 to 7). For example, <C4> drops a piece into column 4, which will automatically fall to the lowest empty row in column 4."""

def construct_observation_prompt(observations):

    legal_actions = observations.get('legal_moves', [])
    opponent_actions = observations.get('opponent_moves', [])
    agent_actions = observations.get('self_moves', [])

    if len(opponent_actions) == 0:
        opponent_prompt = 'Your opponent does not have any move so far.'
    else:
        finished_moves = ','.join(opponent_actions)
        opponent_prompt = f'Your opponent has finished moves: {finished_moves}'

    if len(agent_actions) == 0:
        agent_prompt = 'You do not have any move so far.'
    else:
        finished_moves = ','.join(agent_actions)
        agent_prompt = f'You have finished moves: {finished_moves}'

    if len(legal_actions) == 0:
        legal_position_prompt = 'Currently, it is not your turn to act.'
    else:
        legal_pos = ','.join(legal_actions)
        legal_position_prompt = f'Currently, the legal positions are {legal_pos}.'

    player_idx = observations.get('player_idx', 0)
    symbol = "X (Red)" if player_idx == 0 else "O (Yellow)"
    
    board_str = observations.get('board', '')
    if board_str:
        board_preview = f"The board currently looks like this:\n{board_str}\n"
    else:
        board_preview = ""
        
    prompt = f'You are playing as {symbol}.\n{board_preview}\n{opponent_prompt} {agent_prompt} {legal_position_prompt}'

    return prompt

def _construct_game_history_legend():
    return """- [Player Context]: States which piece ("x" or "o") "You" and "Opponent" correspond to.
- [Position Legend]: Explains how to read the [Position] lines.
- [Position]: The board layout at the moment that player had to make their decision (before the move was executed). The board is a 6x7 grid with cells labeled from C1R1 to C7R6, where rows are 1 (bottom) to 6 (top) and columns are 1 (left) to 7 (right). 'x' and 'o' represent the players' pieces, '.'=empty."""
