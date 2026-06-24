SW_INJECTION_PROMPT = """\
=== YOUR GAME NOTES ===
From your experience in previous games of {game_name}, you have written these notes to guide your future play.

--- GAME NOTES ---
{notes_text}

Use these notes to inform your current move decisions.
=== END GAME NOTES ===
"""

SW_UPDATE_PROMPT = """\
You are reviewing your own experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR CURRENT GAME NOTES ---
{old_notes}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

Based on these {n} recent games and your existing notes, write an updated, general summary of your experience in this game.
These notes will be provided to you in future games to guide your strategy.
"""
