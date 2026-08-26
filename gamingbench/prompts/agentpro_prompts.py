AGENTPRO_INJECTION_PROMPT = """\
=== PREVIOUS GAME REFLECTION ===
As Player {player_index}, your previous reflection on this game against this opponent was:
{ref}
=== END PREVIOUS GAME REFLECTION ===
"""

AGENTPRO_BELIEF_PROMPT = """\
{history_belief}

As Player {player_index}, please analyze your own game situation bracketed with <ses> and </ses>, that includes your current state, feasible game strategies, the game situation of opponents bracketed with <ops> and </ops>, that includes opponents' behavior, opponents' possible strategies, and then briefly talk about your opponent's opinion of you bracketed with <opo> and </opo>, and finally give the most reasonable action in the form of '{{"action": "..."}}'. Note that your action must be selected from {legal_actions}. For example, <ses> My state is ... </ses>, <ops> I think that opponent is ... </ops>, <opo> The opponent thinks that ... </opo>. I will choose {{"action": "{example_action}"}}.
You should use Player {player_index} as the first person.
"""

AGENTPRO_CHAT_BELIEF_PROMPT = """\
{history_belief}

As Player {player_index}, please analyze your own game situation bracketed with <ses> and </ses>, that includes your current state, feasible game strategies, the game situation of opponents bracketed with <ops> and </ops>, that includes opponents' behavior, opponents' possible strategies, and then briefly talk about your opponent's opinion of you bracketed with <opo> and </opo>, and finally write the chat message you wish to send enclosed in <chat> and </chat> tags. For example, <ses> My state is ... </ses>, <ops> I think that opponent is ... </ops>, <opo> The opponent thinks that ... </opo>. <chat> This is my message! </chat>
You should use Player {player_index} as the first person.
"""

AGENTPRO_REFLECTION_PROMPT = """\
You are reviewing your experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- GAME HISTORY ---
{game_history}

In the latest game, you have established a belief like this:
your own game situation is: {ses}
the game situation of opponents is: {ops}
your opponent's opinion of you is: {opo}

Based on the game history, please summarize the reasons for your failure or success bracketed with <rea> and </rea> and briefly propose a reasonable strategy bracketed with <ref> and </ref>. You should use Player {player_index} as the first person.
"""
