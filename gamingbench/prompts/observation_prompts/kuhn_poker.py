def _construct_head_prompt():
    return 'Kuhn poker is a simple zero-sum two-player imperfect-information game, amenable to a complete game-theoretic analysis. In Kuhn poker, the deck includes only three playing cards: a King (K), a Queen (Q), and a Jack (J).\n' \
           'One card is dealt to each player, and the third is put aside unseen. The players take turns either to <Bet> or <Pass>.\n' \
           'If a player bets, the other player must either call the bet by betting or fold by passing. If both players pass, the game is over, and the player with the higher-ranking card wins. The card rankings are: King (K) > Queen (Q) > Jack (J).\n' \
           '\n' \
           'You are playing Kuhn poker with the opponent. The actions are denoted by <Bet> and <Pass>.'



def construct_observation_prompt(observations):

    card_mapping = {
        '0': 'Jack (J)',
        '1': 'Queen (Q)',
        '2': 'King (K)'
    }

    card = card_mapping[observations['card']]
    moves = observations['moves']
    player_idx = observations['player_idx']

    move_prompt = ''
    if moves is not None:
        move_prompt = 'Here are the past moves in this match:\n'

        for idx, m in enumerate(moves):
            action_player_idx = 0 if idx % 2 == 0 else 1
            role = 'you' if action_player_idx == player_idx else 'the opponent'

            if m == 'b':
                move = '<Bet>'
            elif m == 'p':
                move = '<Pass>'
            else:
                raise ValueError

            if idx == 0:
                move_prompt += f'In the {idx + 1}st round, {role} choose to {move};\n'
            elif idx == 1:
                move_prompt += f'In the {idx + 1}nd round, {role} choose to {move};\n'
            elif idx == 2:
                move_prompt += f'In the {idx + 1}rd round, {role} choose to {move};\n'
            else:
                raise ValueError

    prompt = f'In this match, your card is {card}.\n' \
             f'{move_prompt}\n' \
             f'Your legal moves are: <Pass>, <Bet>.'

    player_idx = observations.get("player_idx", 0)
    prompt = f"You are playing as Player {player_idx + 1}.\n" + prompt
    return prompt


if __name__ == '__main__':
    
    obs_prompt = construct_observation_prompt(
        {'card': 0, 'moves': 'pb', 'player_idx': 0})
    prompt += '\n' + obs_prompt
    print(prompt)
