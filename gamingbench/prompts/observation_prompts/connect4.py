

def _construct_head_prompt():
    return """Connect 4 is a two-player connection board game, where the players choose a color and then take turns dropping colored discs into a vertically suspended grid. The pieces fall straight down, occupying the next available space within the column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. You are a gaming agent that aims to beat me in Connect 4 games. 
    Each move is represented by a string consisting of two parts: the column (C) and the row (R), in that order. For instance, C1 means the first column."""

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
- [Position]: The board layout at the moment that player had to make their decision (before the move was executed). The board is a 6x7 grid. 'x' and 'o' represent the players' pieces, '.'=empty."""
