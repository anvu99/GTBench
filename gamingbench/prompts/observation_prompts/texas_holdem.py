def _construct_head_prompt():
    return "Texas Hold'em is a popular poker game. In this Limit Texas Hold'em game, the bet and raise amounts are fixed. " \
           "Each player is dealt two private hole cards, and up to five community cards are dealt face-up on the board.\n" \
           "Cards are represented by a two-character string '{Suit}{Rank}'. Suits are S (Spades), H (Hearts), D (Diamonds), C (Clubs). " \
           "Ranks are 2-9, T (Ten), J (Jack), Q (Queen), K (King), A (Ace). For example, 'ST' means Ten of Spades, 'CQ' means Queen of Clubs.\n" \
           "There are four betting rounds: Preflop, Flop, Turn, and River. During each betting round, players can choose " \
           "their actions from the provided legal actions list.\n" \
           "Your ultimate goal is to maximize your total chips won across many matches by making good decisions based on your hand strength and the board."

def construct_observation_prompt(observations):
    player_idx = observations['player_idx']
    board_str = observations['board_str']
    legal_moves = observations['legal_moves']
    action_record = observations.get('action_record', '')
    
    prompt = f"Your identity is player {player_idx}.\n"
    prompt += f"{board_str}\n"
    prompt += f"The actions you can choose are {legal_moves}.\n"
    if action_record:
        prompt += f"{action_record}\n"
        
    is_chat = observations.get('is_chat_phase', False)
    is_active = observations.get('is_active_player', True)
    
    if is_chat and is_active:
        prompt += '\nYou are currently in the chat phase. Before you make your game move, you can communicate with your opponent. You are generating a chat message.'
    
    prompt += "\nPlease provide your results in the form of {'action': ''}. You must choose one from "
    prompt += f"{legal_moves} as your answer."

    return prompt

def _construct_game_history_legend():
    return ""

