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
Format your output as a bulleted list of observations, with one atomic observation per bullet.
"""

SW_UPDATE_PROMPT_ANTI_DECAY = SW_UPDATE_PROMPT + """\
Never remove a bullet from the previous note. You must carry all past observations forward and only append new ones or update existing ones.
"""

SW_UPDATE_PROMPT_ANTI_CONTRADICTION = SW_UPDATE_PROMPT + """\
Never have two bullets that contradict each other. If a new observation contradicts an existing bullet, remove the old one and keep only the new observation.
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
Format your output as a bulleted list of observations, with one atomic observation per bullet.
"""

SW_OBS_UPDATE_PROMPT_ANTI_DECAY = SW_OBS_UPDATE_PROMPT + """\
Never remove a bullet from the previous note. You must carry all past observations forward and only append new ones or update existing ones.
"""

SW_OBS_UPDATE_PROMPT_ANTI_CONTRADICTION = SW_OBS_UPDATE_PROMPT + """\
Never have two bullets that contradict each other. If a new observation contradicts an existing bullet, remove the old one and keep only the new observation.
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

# =============================================================================
# Mode: "reputation"  (Variant 1)
# Same as SW_UPDATE_PROMPT but WITHOUT the bullet-list requirement.
# The LLM may choose any representation it finds most useful.
# =============================================================================

SW_UPDATE_PROMPT_REPUTATION = """\
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
This note should STRICTLY store observations and insights you have perceived about the opponent — their behavioral tendencies,
patterns, and hidden intent.
DO NOT include any strategy, counter-strategy, recommendations, or instructions on what you should do. Focus EXCLUSIVELY on describing the opponent.
"""

SW_UPDATE_PROMPT_REPUTATION_ANTI_DECAY = SW_UPDATE_PROMPT_REPUTATION + """\
Never remove any insight from the previous note. You must carry all past observations forward
and only append new ones or update existing ones.
"""

SW_UPDATE_PROMPT_REPUTATION_ANTI_CONTRADICTION = SW_UPDATE_PROMPT_REPUTATION + """\
Never include two statements that contradict each other. If a new observation contradicts an
existing one, remove the old one and keep only the new observation.
"""

# =============================================================================
# Mode: "strategy"  (Variant 2)
# The agent is asked to synthesize a personal strategy (what it should do),
# without any explicit instruction to track the opponent's reputation.
# =============================================================================

SW_UPDATE_PROMPT_STRATEGY = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR PREVIOUS STRATEGY ---
{old_notes}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

Based on these {n} recent games, write an updated strategy for yourself to use in future games.
This strategy should STRICTLY contain actionable instructions, rules, and plans for what YOU should do.
DO NOT include any observations, reputation notes, or descriptive analysis of the opponent's behavior. Focus EXCLUSIVELY on generating your own strategy.
"""

# =============================================================================
# Mode: "dual" / "dual-structured"  — Call 1: reputation update
# Inputs: old_notes (previous reputation), game_histories, n, game_name, game_intro
# Output: updated free-form reputation note
# (anti-decay / anti-contradiction flags apply to this call only)
# =============================================================================

SW_UPDATE_PROMPT_DUAL_REPUTATION = SW_UPDATE_PROMPT_REPUTATION

SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_DECAY = SW_UPDATE_PROMPT_REPUTATION_ANTI_DECAY

SW_UPDATE_PROMPT_DUAL_REPUTATION_ANTI_CONTRADICTION = SW_UPDATE_PROMPT_REPUTATION_ANTI_CONTRADICTION

# =============================================================================
# Mode: "dual"  — Call 2: strategy update (free-form)
# Inputs: reputation_notes (NEW, from Call 1), old_strategy, game_histories, n
# =============================================================================

SW_UPDATE_PROMPT_DUAL_STRATEGY = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- UPDATED OPPONENT REPUTATION NOTE ---
{reputation_notes}

--- YOUR PREVIOUS STRATEGY ---
{old_strategy}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

Based on the updated reputation note and these {n} recent games, write an updated strategy
for yourself to use in future games.
Include anything you believe will help you perform better.
"""

# =============================================================================
# Mode: "dual-structured"  — Call 2: strategy update (structured EXPLOIT/AVOID/DEFAULT)
# Same inputs as SW_UPDATE_PROMPT_DUAL_STRATEGY, but enforces the structured output format.
# The OPPONENT: line keeps the WHEN/OPPONENT/DO structure but does not require statistics.
# =============================================================================

SW_UPDATE_PROMPT_DUAL_STRATEGY_STRUCTURED = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- UPDATED OPPONENT REPUTATION NOTE ---
{reputation_notes}

--- YOUR PREVIOUS STRATEGY ---
{old_strategy}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

Based on the updated reputation note and these {n} recent games, produce a comprehensive strategy
brief for playing against this opponent in future games. This brief will be the ONLY opponent
information the agent sees during play — write it to be immediately actionable.

== STRATEGY GENERATION RULES ==

Focus on repeating, predictable opponent patterns observed in the trajectories and reputation note.
For EXPLOIT items: structure as WHEN / OPPONENT / DO.
For AVOID items: structure as WHEN / OPPONENT / DON'T / INSTEAD.
Use the game trajectories to validate which plays worked and which backfired this batch.
Refine the previous strategy: update claims where new evidence changes the picture, keep what is still valid.
When there is no clear signal for a situation, fall back to the most rational play given the game rules.

OPPONENT line: A brief description of the opponent's observed behavior that justifies the DO/INSTEAD
recommendation. Statistics are optional — include them only if the trajectories clearly support them.
  EXAMPLE: "Tends to bid aggressively when holding high-value items" or "Usually defects late-game"

== OUTPUT FORMAT ==

=== STRATEGY BRIEF (Based on {games_observed} games) ===

[EXPLOIT — Do These]
1. WHEN: <the observable trigger situation>
   OPPONENT: <brief description of observed behavior>
   DO: <specific, concrete recommendation — what bid/move to make>
2. <...>

[AVOID — Don't Do These]
1. WHEN: <the observable trigger situation>
   OPPONENT: <brief description of observed behavior>
   DON'T: <the naive/tempting play to avoid>
   INSTEAD: <what to do instead>
2. <...>

[DEFAULT — When No Clear Signal]
<Rational/optimal-play description for situations not covered by the patterns above.>

===
"""

# =============================================================================
# Injection prompts (in-game, per memory mode)
# =============================================================================

# Mode "reputation" — same framing as the original SW_INJECTION_PROMPT
SW_REPUTATION_INJECTION_PROMPT = """\
=== OPPONENT REPUTATION NOTE ===
From your experience in previous games of {game_name}, you have built up this note about your opponent.
It captures insights about their behavioral tendencies, patterns, and hidden intent to help you
anticipate their moves and make better decisions.

--- OPPONENT REPUTATION NOTE ---
{notes_text}

Use this note to inform your current move decisions.
=== END OPPONENT REPUTATION NOTE ==="""\

# Mode "strategy" — shows the strategy brief only
SW_STRATEGY_INJECTION_PROMPT = """\
=== YOUR STRATEGY NOTE ===
From your experience in previous games of {game_name}, you have developed this strategy
for playing against your opponent. Use it to guide your current decisions.

--- STRATEGY NOTE ---
{notes_text}

Use this strategy to inform your current move decisions.
=== END STRATEGY NOTE ==="""\

# Mode "dual" — shows both reputation and strategy as separate blocks
SW_DUAL_INJECTION_PROMPT = """\
=== OPPONENT REPUTATION NOTE ===
From your experience in previous games of {game_name}, you have built up this note about your opponent.
It captures insights about their behavioral tendencies, patterns, and hidden intent.

--- OPPONENT REPUTATION NOTE ---
{reputation_text}

=== END OPPONENT REPUTATION NOTE ===

=== YOUR STRATEGY NOTE ===
Based on your experience, you have developed this strategy for playing against your opponent.

--- STRATEGY NOTE ---
{strategy_text}

Use both of the above to inform your current move decisions.
=== END STRATEGY NOTE ==="""\

# Mode "dual-structured" — shows strategy block only (same as SW_STRATEGY_INJECTION_PROMPT)
SW_DUAL_STRUCTURED_INJECTION_PROMPT = SW_STRATEGY_INJECTION_PROMPT

