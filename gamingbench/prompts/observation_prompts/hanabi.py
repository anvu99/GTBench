def _construct_head_prompt(enable_chat=False, game_config=None):
    if game_config is not None:
        colors = game_config.colors
        ranks = game_config.ranks
        max_life_tokens = game_config.max_life_tokens
        max_score = colors * ranks
        
        colors_names = ['R', 'Y', 'G', 'W', 'B'][:colors] if colors <= 5 else [f"C{i}" for i in range(colors)]
        colors_str = f"{colors} colors ({', '.join(colors_names)})"
        
        comp_counts = []
        for rank_idx in range(ranks):
            if rank_idx == 0:
                comp_counts.append(3)
            elif rank_idx == ranks - 1:
                comp_counts.append(1)
            else:
                comp_counts.append(2)
            
        comp_parts = []
        for i, count in enumerate(comp_counts):
            rank = i + 1
            if count == 1:
                comp_parts.append(f"one {rank}")
            elif count == 2:
                comp_parts.append(f"two {rank}s")
            elif count == 3:
                comp_parts.append(f"three {rank}s")
            else:
                comp_parts.append(f"{count} {rank}s")
                
        if len(comp_parts) > 1:
            comp_str = ", ".join(comp_parts[:-1]) + f", and {comp_parts[-1]}"
        else:
            comp_str = comp_parts[0]
            
        deck_comp_str = f"Deck composition per color: {comp_str}."
        
        base_prompt = f'You are playing Hanabi, a cooperative card game. The team works together to build fireworks of {colors_str} in order from 1 to {ranks}. ' \
               f'{deck_comp_str} ' \
               'CRITICALLY: For each player, they CANNOT see their own hand (meaning they absolutely CANNOT see the color or number/rank of their own cards), but they CAN see all of their teammates\' hands to give them hints. ' \
               'You CANNOT hint yourself, you can only hint teammates. Cards drawn from the deck are completely unknown until hinted. ' \
               'You must use hints to communicate, or play/discard based on hints you receive. '
               
        if enable_chat:
            communication_prompt = 'Direct communication IS enabled via chat, so use it to coordinate before taking your action. '
        else:
            communication_prompt = 'Direct communication is disabled. Your actions are your only way to communicate. '

        max_info = getattr(game_config, 'max_information_tokens', 8)
        rules_prompt = f'On your turn, you MUST take one of three actions: PLAY a card from your hand, DISCARD a card (regains 1 Info token; only allowed if you have less than the maximum {max_info} Info tokens), or HINT a teammate about the color or rank of their cards (costs 1 Info token). ' \
               f'Note: A "Round" in this context refers to a single turn/action taken by one player. Hints on cards will be marked with "@ Round X" to indicate the exact round the hint was generated. ' \
               f'You can use information on when each card was drawn (e.g., "(Drawn Round 2)"), when hints were given, and the "Timeline of hints" to help your logical deductions. ' \
               f'To play successfully, a card MUST match the EXACT next needed rank for its color stack (e.g., you MUST play a 2 on a 1 stack). Playing a duplicate or wrong rank will fail. ' \
               f'A failed play costs 1 Life token and discards the card. Successfully playing a {ranks} restores 1 Info token. ' \
               'When giving a HINT, you must point out ALL cards of that color or rank in the player\'s hand. ' \
               f'The game ends immediately if {max_life_tokens} Life tokens are lost, the score reaches {max_score}, or everyone gets one final turn after the deck empties. ' \
               'Your team score is the total number of cards successfully played across all colors. You must work together to maximize this score as much as possible.'
    else:
        base_prompt = 'You are playing Hanabi, a cooperative card game. The team works together to build fireworks of 5 colors (R, Y, G, W, B) in order from 1 to 5. ' \
               'Deck composition per color: three 1s, two 2s, two 3s, two 4s, and only one 5. ' \
               'CRITICALLY: For each player, they CANNOT see their own hand (meaning they absolutely CANNOT see the color or number/rank of their own cards), but they CAN see all of their teammates\' hands to give them hints. ' \
               'You CANNOT hint yourself, you can only hint teammates. Cards drawn from the deck are completely unknown until hinted. ' \
               'You must use hints to communicate, or play/discard based on hints you receive. '
               
        if enable_chat:
            communication_prompt = 'Direct communication IS enabled via chat, so use it to coordinate before taking your action. '
        else:
            communication_prompt = 'Direct communication is disabled. Your actions are your only way to communicate. '

        rules_prompt = 'On your turn, you MUST take one of three actions: PLAY a card from your hand, DISCARD a card (regains 1 Info token; only allowed if you have less than the maximum 8 Info tokens), or HINT a teammate about the color or rank of their cards (costs 1 Info token). ' \
               'Note: A "Round" in this context refers to a single turn/action taken by one player. Hints on cards will be marked with "@ Round X" to indicate the exact round the hint was generated. ' \
               'You can use information on when each card was drawn (e.g., "(Drawn Round 2)"), when hints were given, and the "Timeline of hints" to help your logical deductions. ' \
               'To play successfully, a card MUST match the EXACT next needed rank for its color stack (e.g., you MUST play a 2 on a 1 stack). Playing a duplicate or wrong rank will fail. ' \
               'A failed play costs 1 Life token and discards the card. Successfully playing a 5 restores 1 Info token. ' \
               'When giving a HINT, you must point out ALL cards of that color or rank in the player\'s hand. ' \
               'The game ends immediately if 3 Life tokens are lost, the score reaches 25, or everyone gets one final turn after the deck empties. ' \
               'Your team score is the total number of cards successfully played across all colors. You must work together to maximize this score as much as possible.'
               
    return base_prompt + communication_prompt + rules_prompt

def construct_observation_prompt(observations):
    player_idx = observations['player_idx']
    num_players = observations['num_players']
    
    fw = observations['fireworks']
    fw_str = ", ".join([f"{k}={v}" for k, v in fw.items()])
    
    status_str = f"Round: {observations.get('round_num', '?')}\n" \
                 f"Fireworks: {fw_str}\n" \
                 f"Tokens -> Life: {observations['life_tokens']} | Info: {observations['info_tokens']}\n" \
                 f"Deck size: {observations['deck_size']} | Discard pile: {', '.join(observations['discard_pile'])}\n"
                 
    own_hand_str = f"--- Your Hand (Player {player_idx}) ---\n"
    own_hand_str += f"CRITICAL REMINDER: You are Player {player_idx}. You CANNOT see your own cards, only the hints you have received.\n"
    
    own_hist = observations.get('hint_histories', {}).get(player_idx, [])
    if own_hist:
        own_hand_str += f"Timeline of hints you received: {', '.join(own_hist)}\n"
    else:
        own_hand_str += f"Timeline of hints you received: None\n"
        
    for i, hints in enumerate(observations['own_card_knowledge']):
        own_hand_str += f"Card {i}: {hints}\n"
        
    other_hands_str = "--- Teammates' Hands (Visible to you) ---\n"
    other_hands_str += "CRITICAL REMINDER: These are your teammates' actual cards. YOUR TEAMMATES CANNOT SEE THESE CARDS (meaning they CANNOT see the color or number/rank of these cards)! They ONLY know the information listed in the [hints: ...] brackets.\n"
    for p_idx, hand in observations['other_hands'].items():
        other_hands_str += f"Player {p_idx}:\n"
        other_hist = observations.get('hint_histories', {}).get(p_idx, [])
        other_hands_str += f"Timeline of hints they received: {', '.join(other_hist) if other_hist else 'None'}\n"
        for i, card_str in enumerate(hand):
            other_hands_str += f"Card {i}: {card_str}\n"
        
    legal_moves = observations['legal_moves']
    legal_str = "\n".join(legal_moves)
    
    prompt = f"You are playing as Player {player_idx} of {num_players}.\n\n" + \
             status_str + "\n" + own_hand_str + "\n" + other_hands_str + \
             "\n--- Legal Actions ---\n" + legal_str
             
    return prompt

def _construct_game_history_legend():
    return """\
- [State]: A snapshot of the fireworks score, life tokens, info tokens, and deck size at the start of the round.
- [Hands]: The cards held by each player at that moment, along with the hints they had received about them. NOTE: The hand of the player whose turn it is (marked "acting") is intentionally hidden as 'Unknown' to reflect what they could see at that moment—they only know the information in the [hints: ...] brackets! Hints include an "@ Round X" tag to indicate the exact round (turn) they were received.
- [Move]: The action taken by the player. <PLAY N> plays the Nth card, <DISCARD N> discards it. <HINT PLAYER N COLOR/RANK> reveals matching cards in that player's hand.
- [Outcome]: Whether a PLAY was SUCCESS (added to fireworks) or FAIL (lost a life token), and the token impacts of DISCARD or HINT.\
"""
