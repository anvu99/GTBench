EW_INJECTION_PROMPT = """\
=== OPPONENT REPUTATION NOTE ===
From your experience in previous games of {game_name}, you have built up this note about your opponent. It captures insights about their behavioral tendencies, patterns, and hidden intent to help you anticipate their moves and make better decisions.

--- OPPONENT REPUTATION NOTE ---
{notes_text}

Use this note to inform your current move decisions.
=== END OPPONENT REPUTATION NOTE ==="""

EW_ENDGAME_OBS_PROMPT = """\
You are reviewing your experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- GAME HISTORY ---
{game_history}

In no more than 10 sentences, summarize what happened in this game that you want to remember for future game.
"""

EW_WINDOW_SYNTHESIS_PROMPT = """\
You are reviewing your experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- RECENT GAME TRAJECTORIES (last {n} games) ---
{observations}

Based on these {n} game trajectories, write an updated Opponent Reputation Note.
This note should store insights you have perceived about the opponent — their behavioral tendencies, patterns, and hidden intent — that can help you anticipate their moves and perform better in future games.
"""
