EW_INJECTION_PROMPT = """\
=== YOUR GAME NOTES ===
From your experience in previous games of {game_name}, you have written these notes to guide your future play.

--- GAME NOTES ---
{notes_text}

Use these notes to inform your current move decisions.
=== END GAME NOTES ===
"""

EW_ENDGAME_OBS_PROMPT = """\
You are reviewing your experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- GAME HISTORY ---
{game_history}

In no more than 4 sentences, summarize what happened in this game that you want to remember for future game.
"""

EW_WINDOW_SYNTHESIS_PROMPT = """\
You are reviewing your experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR RECENT GAME OBSERVATIONS (last {n} games) ---
{observations}

Based on these {n} observations, write general notes to guide your strategy in future games of {game_name}. These notes will be provided to you at the start of each future game.
"""
