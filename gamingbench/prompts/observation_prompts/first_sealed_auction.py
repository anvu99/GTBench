
def _construct_head_prompt():
    return 'A first-price sealed-bid auction (FPSBA) is a common type of auction. It is also known as blind auction. ' \
           'In this type of auction, bidders generate their bids one by one, keeping their bids secret from each other. ' \
           'The highest bidder wins the object and pays the price that was submitted. ' \
           'CRITICALLY: Your goal is NOT just to win the object, but to maximize your profit (Utility). ' \
           'If you win the auction, your utility for this round is calculated as (Your Budget - Your Bid). ' \
           'If you lose the auction, your utility is 0. ' \
           'If both players bid the same amount, it is a tie and both players receive 0 utility. ' \
           'You will play multiple matches against this opponent, and your cumulative utility across all matches is your final score. ' \
           'Therefore, you must carefully balance the probability of winning against the profit margin of your bid.' \
           '\n' \
           'Each action is represented by <x> where x refers to the bid.'


def construct_observation_prompt(observations):

    valuation = observations['valuation']
    is_chat = observations.get('is_chat_phase', False)
    is_active = observations.get('is_active_player', True)

    prompt = f'Now, you are in an auction with an opponent. Your budget (valuation) for the object is {valuation}. Your bid must be strictly lower than or equal to {valuation}. ' \
             f'You shall bid wisely against your opponent to maximize your expected utility. \n' \
             f'Your opponent also has a private budget randomly drawn from the uniform distribution [1, 10] and you do not know it.' \
             f'\n\n'

    self_moves = observations.get('self_moves', [])
    game_round = observations.get('game_round', 1)

    if is_chat:
        if is_active:
            if game_round == 1:
                prompt += 'You are currently in the chat phase. You are generating a chat message before you submit your bid.'
            else:
                prompt += 'You are currently in the chat phase. Your opponent has already submitted their bid, but it is hidden from you. You are generating a chat message before you submit your bid.'
        else:
            if game_round == 1:
                prompt += 'You are currently in the chat phase. It is your opponent\'s turn to make a bid, and their bid will be hidden from you. You are generating a chat message before your opponent submits their bid.'
            else:
                prompt += f'You are currently in the chat phase. It is your opponent\'s turn to make a bid, and their bid will be hidden from you. You have already submitted your bid in the previous round (Your bid was: {", ".join(self_moves)}). You are generating a chat message before your opponent submits their bid.'
    else:
        if game_round == 1:
            prompt += 'It is your turn to bid. Your opponent will bid in the next round.'
        else:
            prompt += 'It is your turn to bid. Your opponent has already submitted their bid, but it is hidden from you.'
        
    legal_moves = observations.get('legal_moves', [])
    legal_move_str = ', '.join(legal_moves)
    prompt += f'\nThe legal actions are: {legal_move_str}.'
        
    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt
