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

SW_OBS_UPDATE_PROMPT = """\
You are reviewing your own experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR CURRENT OPPONENT REPUTATION NOTE ---
{old_notes}

--- REFINED OBSERVATIONS FROM LATEST BATCH ({n} games) ---
Below are your final refined observations from the latest {n} games against the same opponent:

{game_histories}

Based on these {n} recent observations, write an updated Opponent Reputation Note.
This note should store insights you have perceived about the opponent — their behavioral tendencies, patterns, and hidden intent — that can help you anticipate their moves and perform better in future games.
"""

SW_OBS_GENERATION_SUFFIX = """\
Additionally, after your move, write a concise in-game observation inside <obs>...</obs> tags.
Summarize what you have observed about your opponent's strategy and behavior so far in THIS game.
Keep it under 5 sentences. This observation will carry forward to your next turn."""

SW_OBS_INJECTION_PROMPT = """\
=== YOUR IN-GAME OBSERVATION (This Game) ===
This is your own running observation about your opponent, updated from previous turns in this game.
Use it to reason about their current strategy.

{in_game_obs}
=== END IN-GAME OBSERVATION ==="""

SW_OBS_FINAL_REFINEMENT_PROMPT = """\
You just finished a game of {game_name}.

--- YOUR RUNNING IN-GAME OBSERVATION ---
{in_game_obs}

--- FULL GAME TRAJECTORY ---
{game_history}

Based on the complete trajectory, write a final refined observation about your opponent.
Focus on their behavioral tendencies, patterns, and hidden intent.
Keep it concise (under 10 sentences)."""
