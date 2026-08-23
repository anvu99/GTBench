SW_INJECTION_PROMPT = """\
=== OPPONENT REPUTATION NOTE ===
From your experience in previous games of {game_name}, you have built up this note about your opponent. It captures insights about their behavioral tendencies, patterns, and hidden intent to help you anticipate their moves and make better decisions.

--- OPPONENT REPUTATION NOTE ---
{notes_text}

Use this note to inform your current move decisions.
=== END OPPONENT REPUTATION NOTE ==="""

SW_UPDATE_PROMPT = """\
You are reviewing your own experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR CURRENT OPPONENT REPUTATION NOTE ---
{old_notes}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

Based on these {n} recent games, write an updated Opponent Reputation Note.
This note should store insights you have perceived about the opponent — their behavioral tendencies, patterns, and hidden intent — that can help you anticipate their moves and perform better in future games.
"""
