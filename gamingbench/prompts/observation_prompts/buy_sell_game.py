
def _construct_head_prompt():
    return (
        'You are playing a bilateral negotiation game between a Seller and a Buyer. '
        'One player is the Seller, who owns one unit of an item. '
        'The other player is the Buyer, who wants to purchase that item. '
        'Each player has a PRIVATE value: '
        'the Seller has a private cost of production (the minimum price they are willing to accept), '
        'and the Buyer has a private willingness-to-pay (the maximum price they are willing to pay). '
        'These values are NOT revealed to the other player — you only know your own. '
        'Players alternate turns proposing a price, starting with the Seller. '
        'A deal is made when one player accepts the other\'s proposal. '
        'SCORING: If a deal is reached at price P, '
        'the Seller earns (P - cost) and the Buyer earns (WTP - P). '
        'If no deal is reached (REJECT or no agreement by the end of all rounds), both players score 0. '
        'You will play multiple matches against this opponent, and your cumulative score across all matches is your final score. '
        'Therefore, negotiate strategically: do not reveal your private valuation unless it benefits you. '
        'Each action must be one of: '
        '<PROPOSE: X> to propose a price of X dollars, '
        '<ACCEPT> to accept the most recent offer from your opponent, or '
        '<REJECT> to end negotiations immediately (both score 0).'
    )


def _construct_game_history_legend():
    return (
        '[Position Legend] Each [Position] line shows the game state before that player\'s move. '
        'Format: Role, private valuation, current turn, and last offer on the table.\n'
        '[Move] lines show the action taken (PROPOSE, ACCEPT, or REJECT).\n\n'
    )


def construct_observation_prompt(observations):
    player_idx = observations.get('player_idx', 0)
    role = observations.get('player_role', 'Seller')
    private_val = observations.get('private_valuation')
    last_price = observations.get('last_proposed_price')
    last_proposer_role = observations.get('last_proposer_role')
    turn = observations.get('turn_number', 1)
    max_turns = observations.get('max_turns', 10)
    legal_moves = observations.get('legal_moves', [])

    # Role-specific private info
    if role == 'Seller':
        val_prompt = (
            f'You are the Seller. Your private cost of production is {private_val}. '
            f'This is the minimum price you should accept to avoid a loss. '
            f'Do NOT reveal your cost to the Buyer unless strategically beneficial.'
        )
    else:
        val_prompt = (
            f'You are the Buyer. Your private willingness-to-pay is {private_val}. '
            f'This is the maximum price you should pay. '
            f'Do NOT reveal your WTP to the Seller unless strategically beneficial.'
        )

    # Current offer on the table
    if last_price is None:
        offer_prompt = 'There is no offer on the table yet.'
    else:
        offer_prompt = (
            f'The most recent offer on the table is ${last_price}, '
            f'proposed by the {last_proposer_role}.'
        )
        if last_proposer_role != role:
            offer_prompt += ' You may ACCEPT this offer, propose a counter-offer, or REJECT.'
        else:
            offer_prompt += ' You are waiting for the opponent\'s response, but now it is your turn to make a new proposal or REJECT.'

    # Turn counter
    turns_left = max_turns - turn + 1
    turn_prompt = (
        f'This is Turn {turn} of {max_turns}. '
        f'There are {turns_left} turn(s) remaining (including this one). '
        'If no deal is reached by the final turn, both players score 0.'
    )

    # Legal moves — hardcoded string since we pass [] to bypass PromptAgent strict matching
    move_prompt = (
        'Your available actions are: <PROPOSE: X>, <ACCEPT>, <REJECT>. '
        'For PROPOSE, replace X with any integer you wish to offer '
        '(e.g., <PROPOSE: 55>, <PROPOSE: 42>).'
    )

    prompt = (
        f'You are playing as Player {player_idx + 1} (the {role}).\n'
        f'{val_prompt}\n\n'
        f'{offer_prompt}\n'
        f'{turn_prompt}\n\n'
        f'{move_prompt}'
    )
    return prompt
