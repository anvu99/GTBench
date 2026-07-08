
def _construct_head_prompt():
    return 'A first-price sealed-bid auction (FPSBA) is a common type of auction. It is also known as blind auction. ' \
           'In this type of auction, all bidders simultaneously submit sealed bids so that no bidder knows the bid of any other participant. ' \
           'The highest bidder wins the object and pays the price that was submitted. ' \
           'CRITICALLY: Your goal is NOT just to win the object, but to maximize your profit (Utility). ' \
           'If you win the auction, your utility for this round is calculated as (Your Budget - Your Bid). ' \
           'If you lose the auction, your utility is 0. ' \
           'You will play multiple matches against this opponent, and your cumulative utility across all matches is your final score. ' \
           'Therefore, you must carefully balance the probability of winning against the profit margin of your bid.' \
           '\n' \
           'Each action is represented by <x> where x refers to the bid.'


def construct_observation_prompt(observations):

    legal_moves = observations['legal_moves']
    valuation = observations['valuation']
    legal_move_str = ', '.join(legal_moves)

    prompt = f'Now, you are in an auction with an opponent. Your budget (valuation) for the object is {valuation}. Your bid must be strictly lower than or equal to {valuation}. ' \
             f'You shall bid wisely against your opponent to maximize your expected utility. \n' \
             f'Your opponent also has a private budget and you do not know it.' \
             f'\n\n' \
             f'The legal actions are: {legal_move_str}. '
    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt
