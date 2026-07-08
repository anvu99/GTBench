def _construct_head_prompt(enable_chat=False):
    base_prompt = 'You are negotiating the division of Peppers, Strawberries, and Cherries with the opponent. Different items hold different values for both you and your opponent. ' \
           'CRITICALLY: This is a COOPERATIVE negotiation. The other player is your ALLY. Your ultimate goal is to maximize the JOINT UTILITY (the sum of both your score and your partner\'s score) across multiple matches. '
           
    if enable_chat:
        communication_prompt = 'Since direct communication IS enabled, you should use the chat channel to explicitly coordinate and share information before locking in your formal <Proposal> or <Utterance>. '
    else:
        communication_prompt = 'Since direct communication is disabled, you must use your <Proposal> and <Utterance> actions as implicit signals to deduce each other\'s private valuations and find the perfect Pareto-optimal split where no items are wasted. '

    rules_prompt = 'If an agreement is reached, your TRUE SCORE for the match is the JOINT UTILITY: the sum of your individual score plus your partner\'s individual score (where individual score is the items received multiplied by private valuations). ' \
           'If negotiations break down and no agreement is reached, both of your scores are 0. ' \
           'The process is structured into two stages per round: the proposal stage and the utterance stage.'
           
    return base_prompt + communication_prompt + rules_prompt

def _construct_propose_stage_prompt():
    return 'Now, you are in the Proposal stage: you\'ll determine the division of items you desire. This is expressed as [a, b, c], where \'a\' represents the quantity of Peppers, \'b\' the quantity of Strawberries, and \'c\' the quantity of Cherries you wish to acquire. It\'s crucial to base this division on your private valuations and your deduction of your partner\'s valuations. You must reach a mutually agreeable solution to avoid a score of 0, and you should cooperatively aim to maximize the JOINT value extracted from the split. WARNING: Your Proposal is a binding offer. If the opponent replies with <Agree>, the game ends immediately and the items are split exactly according to your Proposal.'

def _construct_utterance_stage_prompt(enable_chat=False):
    if enable_chat:
        return 'Now, you are in the Utterance Stage: you communicate to your opponent what you want, again in the format [a, b, c]. This utterance is your structural formalization of the deal. Because this is a cooperative game WITH chat, you should have already used the chat channel to align with your partner. Your Utterance is non-binding "cheap talk", but it formalizes your intent. The opponent cannot <Agree> to your Utterance to end the game.'
    else:
        return 'Now, you are in the Utterance Stage: you communicate to your opponent what you want, again in the format [a, b, c]. This utterance is your strategic communication. Because this is a cooperative game without chat, you should use this Utterance to implicitly signal your high-value items to your partner, or to acknowledge their signals. Your Utterance is non-binding "cheap talk". The opponent cannot <Agree> to your Utterance to end the game.'

def _solution_prompt():
    return 'Remember, the key in cooperative negotiations is understanding that your partner also has their value system for these items, which is unknown to you. You must use your proposals and utterances to signal your true desires and help your partner gain a favorable outcome. It\'s important to be adaptive, as the negotiation progresses and you gather more information about your partner\'s preferences.'

def construct_observation_prompt(observations):
    turn_type = observations['turn_type']
    item_pool = observations['item_pool']
    most_recent_proposal = observations['most_recent_proposal']
    most_recent_utterance = observations['most_recent_utterance']
    value_vector = observations['self_value_vector']

    item_pool_prompt = f'There are {item_pool[0]} peppers, {item_pool[1]} strawberries, and {item_pool[2]} cherries in the item pool.'

    value_vector = f'The value of each pepper is {value_vector[0]} for you. The value of each strawberry is {value_vector[1]} for you. ' \
                   f'The value of each cherry is {value_vector[2]} for you.'

    if turn_type == 'Proposal':
        if most_recent_utterance is not None:
            last_utterance_prompt = f'Last time, the utterance of the opponent was to take ' \
                                    f'{most_recent_utterance[0]} peppers, {most_recent_utterance[1]} strawberries, ' \
                                    f'and {most_recent_utterance[2]} cherries from the item pool.'
        else:
            last_utterance_prompt = ''

        if most_recent_proposal is not None:
            last_proposal_prompt = f'Now, the opponent propose to take {most_recent_proposal[0]} peppers, ' \
                                   f'{most_recent_proposal[1]} strawberries, and {most_recent_proposal[2]} cherries from the item pool.'
        else:
            last_proposal_prompt = ''

        stage_prompt = _construct_propose_stage_prompt()
        last_situation_prompt = '\n' + last_proposal_prompt + '\n' + last_utterance_prompt
        query_prompt = 'Now, it is your decision. ' \
                       'If you find the proposal raised by the opponent is acceptable for both of you, you should output <Agree>. ' \
                       'Otherwise, you should output your proposal in the format <Proposal: [a, b, c]>.'

    elif turn_type == 'Utterance':
        if most_recent_utterance is not None:
            last_utterance_prompt = f'Last time, the utterance of the opponent was to take ' \
                                    f'{most_recent_utterance[0]} peppers, {most_recent_utterance[1]} strawberries, ' \
                                    f'and {most_recent_utterance[2]} cherries from the item pool.'
        else:
            last_utterance_prompt = ''

        if most_recent_proposal is not None:
            last_proposal_prompt = f'You proposed to take {most_recent_proposal[0]} peppers, ' \
                                   f'{most_recent_proposal[1]} strawberries, and {most_recent_proposal[2]} cherries from the item pool.'
        else:
            last_proposal_prompt = ''

        enable_chat = observations.get('chat_enabled', False)
        stage_prompt = _construct_utterance_stage_prompt(enable_chat=enable_chat)
        last_situation_prompt = _construct_propose_stage_prompt() + '\n' + last_utterance_prompt + '\n' + last_proposal_prompt
        query_prompt = 'Now, it is your turn to provide your utterance regarding the division of items. ' \
                       'You should output your utterance in the format <Utterance: [a, b, c]>.\n' \
                       'For each category, you can not take all the items in a category, i.e., you can not take all 5 Peppers, 5 Strawberries, or 5 Cherries. ' \
                       'Instead, you have to leave at least one item for each category to your opponent.'
    else:
        raise ValueError

    player_idx = observations.get("player_idx", 0)
    stage_prompt = f"You are playing as Player {player_idx + 1}.\n" + stage_prompt
    return stage_prompt + '\n' + item_pool_prompt + '\n' + value_vector + '\n' + last_situation_prompt + '\n' + query_prompt

if __name__ == '__main__':
    observation = {
        'turn_type': 'Proposal',
        'item_pool': '5 5 5'.split(' '),
        'most_recent_proposal': '2 1 4'.split(' '),
        'most_recent_utterance': '1 3 3'.split(' '),
        'self_value_vector': '6 5 1'.split(' ')
    }
    print(_construct_head_prompt(enable_chat=False) + '\n\n' + construct_observation_prompt(observation))
