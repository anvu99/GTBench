
def _construct_head_prompt():
    return 'You are negotiating the division of Peppers, Strawberries, and Cherries with the opponent. Different items hold different values for both you and your opponent. ' \
           'CRITICALLY: Both you and your opponent have exactly 20 points in total to distribute across the 3 item types. The valuation for each item type is randomly generated between 1 and 18, ensuring the sum of your 3 item valuations always equals exactly 20 points. Your opponent\'s 20-point distribution is unknown to you.' \
           'Your goal is NOT just to reach a deal, but to maximize your profit (Utility). ' \
           'If an agreement is reached, your score for the match is the sum of the items you receive multiplied by your private valuation for those items. ' \
           'If negotiations break down and no agreement is reached, both you and your opponent will get 0 score. ' \
           'You will play multiple matches against this opponent, and your cumulative utility across all matches is your final score. ' \
           'The process is structured into two stages per round: the proposal stage and the utterance stage.' \

def _construct_propose_stage_prompt():
    return 'Now, you are in the Proposal stage: you\'ll determine the division of items you desire. This is expressed as [a, b, c], where \'a\' represents the quantity of Peppers, \'b\' the quantity of Strawberries, and \'c\' the quantity of Cherries you wish to acquire. It\'s crucial to base this division on your private valuations. You must reach a mutually agreeable solution to avoid a score of 0, but you should aggressively aim to extract as much value as possible from the split. WARNING: Your Proposal is a binding offer. If the opponent replies with <Agree>, the game ends immediately and the items are split exactly according to your Proposal.'

def _construct_utterance_stage_prompt():
    return 'Now, you are in the Utterance Stage: you communicate to your opponent what you want, again in the format [a, b, c]. This utterance is your strategic communication and doesn\'t necessarily have to reflect your actual desires or the proposal you formulated in the first stage. It\'s a tool for negotiation, potentially used to mislead, bluff, or strategically reveal information to your opponent. Your Utterance is non-binding "cheap talk". The opponent cannot <Agree> to your Utterance to end the game.'

def _solution_prompt():
    return 'Remember, the key in such negotiations is understanding that your opponent also has their value system for these items, which is unknown to you. Balancing between revealing your true desires and misleading your opponent to gain a favorable outcome is essential. It\'s also important to be adaptive, as the negotiation progresses and you gather more information about your opponent\'s preferences and tactics.'


def construct_observation_prompt(observations):

    turn_type = observations['turn_type']
    item_pool = observations['item_pool']
    most_recent_proposal = observations['most_recent_proposal']
    most_recent_utterance = observations['most_recent_utterance']
    value_vector = observations['self_value_vector']
    is_chat = observations.get('is_chat_phase', False)
    is_active = observations.get('is_active_player', True)

    item_pool_prompt = f'There are {item_pool[0]} peppers, {item_pool[1]} strawberries, and {item_pool[2]} cherries in the item pool.'

    value_vector = f'The value of each pepper is {value_vector[0]} for you. The value of each strawberry is {value_vector[1]} for you. ' \
                   f'The value of each cherry is {value_vector[2]} for you.'

    if turn_type == 'Proposal':
        prop_label = "Opponent's Proposal" if is_active else "Your Proposal"
        utt_label = "Opponent's Utterance" if is_active else "Your Utterance"
        
        if most_recent_utterance is not None:
            last_utterance_prompt = f'Last time, {utt_label} was to take ' \
                                    f'{most_recent_utterance[0]} peppers, {most_recent_utterance[1]} strawberries, ' \
                                    f'and {most_recent_utterance[2]} cherries from the item pool.'
        else:
            last_utterance_prompt = ''

        if most_recent_proposal is not None:
            last_proposal_prompt = f'Now, {prop_label} is to take {most_recent_proposal[0]} peppers, ' \
                                   f'{most_recent_proposal[1]} strawberries, and {most_recent_proposal[2]} cherries from the item pool.'
        else:
            last_proposal_prompt = ''

        query_prompt = 'Now, it is your decision. ' \
                       'If you find the proposal raised by the opponent is acceptable, you should output <Agree>. ' \
                       'Otherwise, you should output your proposal in the format <Proposal: [a, b, c]>.'
        
        if is_chat:
            if is_active:
                stage_prompt = f"You are currently in the {turn_type} stage. Before you make your game move, you can communicate with your opponent. You are generating a chat message."
            else:
                stage_prompt = f"You are currently in the {turn_type} stage. Even though it is your opponent's turn to make a game move, you can communicate with them. You are generating a chat message."
        else:
            stage_prompt = _construct_propose_stage_prompt()

        last_situation_prompt = '\n' + last_proposal_prompt + '\n' + last_utterance_prompt

    elif turn_type == 'Utterance':
        prop_label = "Your Proposal" if is_active else "Opponent's Proposal"
        utt_label = "Opponent's Utterance" if is_active else "Your Utterance"
        
        if most_recent_utterance is not None:
            last_utterance_prompt = f'Last time, {utt_label} was to take ' \
                                    f'{most_recent_utterance[0]} peppers, {most_recent_utterance[1]} strawberries, ' \
                                    f'and {most_recent_utterance[2]} cherries from the item pool.'
        else:
            last_utterance_prompt = ''

        if most_recent_proposal is not None:
            last_proposal_prompt = f'Now, {prop_label} is to take {most_recent_proposal[0]} peppers, ' \
                                   f'{most_recent_proposal[1]} strawberries, and {most_recent_proposal[2]} cherries from the item pool.'
        else:
            last_proposal_prompt = ''

        query_prompt = 'Now, it is your turn to provide your utterance regarding the division of items. The utterance is what you' \
                       'want to told to your opponent and does not mean your real intent. You should output your utterance in the format <Utterance: [a, b, c]>.\n' \
                       'For each category, you can not take all the items in a category, i.e., you can not take all 5 Peppers, 5 Strawberries, or 5 Cherries. ' \
                       'Instead, you have to leave at least one item for each category to your opponent.'
                       
        if is_chat:
            if is_active:
                stage_prompt = f"You are currently in the {turn_type} stage. Before you make your game move, you can communicate with your opponent. You are generating a chat message."
            else:
                stage_prompt = f"You are currently in the {turn_type} stage. Even though it is your opponent's turn to make a game move, you can communicate with them. You are generating a chat message."
            last_situation_prompt = '\n' + last_utterance_prompt + '\n' + last_proposal_prompt
        else:
            stage_prompt = _construct_utterance_stage_prompt()
            last_situation_prompt = _construct_propose_stage_prompt() + '\n' + last_utterance_prompt + '\n' + last_proposal_prompt
    else:
        raise ValueError

    player_idx = observations.get("player_idx", 0)
    stage_prompt = f"You are playing as Player {player_idx + 1}.\n" + stage_prompt
    return stage_prompt + '\n' + item_pool_prompt + '\n' + value_vector + '\n' + last_situation_prompt + ('\n' + query_prompt if query_prompt else '')

if __name__ == '__main__':
    observation = {
        'turn_type': 'Proposal',
        'item_pool': '5 5 5'.split(' '),
        'most_recent_proposal': '2 1 4'.split(' '),
        'most_recent_utterance': '1 3 3'.split(' '),
        'self_value_vector': '6 5 1'.split(' ')
    }
    print(_construct_head_prompt() + '\n\n' + construct_observation_prompt(observation))