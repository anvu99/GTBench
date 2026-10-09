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

# =============================================================================
# Mode: "best_worst" trajectory selection
# Receives winning and losing trajectories in separate sections.
# The LLM must extract different types of observations from each set.
# =============================================================================

SW_UPDATE_PROMPT_BEST_WORST = """\
You are reviewing your own experience in {game_name}.

--- GAME RULES ---
{game_intro}

--- YOUR CURRENT OPPONENT REPUTATION NOTE ---
{old_notes}

--- WINNING GAMES ({n_win} best-scoring from this batch) ---
{winning_histories}

--- LOSING GAMES ({n_lose} worst-scoring from this batch) ---
{losing_histories}

Based on both sets of games, write an updated Opponent Reputation Note.

You must extract insights by applying strict differential analysis between winning and losing games:
- [W] WINNING OBSERVATIONS: Identify the exact sequence of events, offers, or chat messages that triggered the opponent to concede or yield maximum value. Be highly specific about the cause-and-effect.
- [L] LOSING OBSERVATIONS: You must isolate the TRUE differentiator. A behavior or strategy is ONLY a valid [L] observation if it caused a loss AND did NOT occur in the winning games. Identify the unique trigger, opponent counter-strategy, or mistake that derailed this specific game.

Tag each observation with [W] or [L] to indicate its source.
Format your output as a bulleted list of observations, with one atomic observation per bullet.
"""

SW_BEST_WORST_INJECTION_PROMPT = """\
=== OPPONENT REPUTATION NOTE ===
From your experience in previous games of {game_name}, you have built up this note about your opponent.

Observations tagged [W] reveal the exact sequences and triggers that successfully exploit the opponent. Use these as your primary gameplan.
Observations tagged [L] record the unique triggers, opponent counter-strategies, or mistakes that derailed previous games. Use them to avoid falling back into previous failure modes.

--- OPPONENT REPUTATION NOTE ---
{notes_text}

Use this note to inform your current move decisions.
=== END OPPONENT REPUTATION NOTE ==="""

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
# Flag: --sw-track-frequency
# Appended to any memory-update prompt to enable per-observation occurrence counts.
# Works independently of which base mode is used, but is primarily designed for
# use with --sw-anti-decay so that accumulated counts persist across batches.
#
# Counting rule (3 cases):
#   NEW observation        → initialize count at 1
#   UPDATE or KEEP + SAW  → increment count by 1
#   KEEP + DID NOT SEE    → leave count unchanged
# =============================================================================

SW_TRACK_FREQUENCY_SUFFIX = """\
[FREQUENCY TRACKING MODE]
Your existing observations above are numbered. You must track the frequency of these behaviors across the new batch of games.

Output ONLY a JSON object in exactly this format:
```json
{{
  "game_evaluations": [
    {{
      "game_index": 1,
      "supported_ids": [1, 3]
    }},
    {{
      "game_index": 2,
      "supported_ids": [1]
    }}
  ],
  "new_observations": ["New pattern description here"],
  "reworded": {{"2": "Updated wording for observation 2"}}
}}
```

Rules:
1. "game_evaluations": You must include one object for EVERY game in the batch. For each game, list the IDs of the existing observations that were clearly exhibited in that specific game.
2. "new_observations": A list of new, atomic behavioral patterns not covered by existing IDs. Use [] if none.
3. "reworded": A mapping of ID to updated text, used to refine an observation's wording without losing its count. Use {{}} if none.
4. Do NOT output raw observation lists or markdown text outside the JSON block. Let the backend manage the raw numbers.
"""

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
# Mode: "dual" + --sw-explore-strategy  — Call 2: strategy update with exploration
# Ported from TENDENCY_STRATEGY_EXPLORE_PROMPT (SimpleTendencyAgent explore mode):
#   - shows best-performing strategy + current strategy with avg utility
#   - guided 6-step thinking process (understand / diagnose / explore / draft /
#     evaluate / decide)
#   - [INHERIT BEST] protocol for version continuity
# Output stays FREE-FORM (same as SW_UPDATE_PROMPT_DUAL_STRATEGY) so the only
# difference from the plain dual baseline is the exploration mechanism.
# Inputs: game_name, game_intro, reputation_notes, old_strategy, game_histories, n,
#         best_strategy, best_avg_utility, best_games_played,
#         current_avg_utility, current_games_played
# =============================================================================

SW_UPDATE_PROMPT_DUAL_STRATEGY_EXPLORE = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- UPDATED OPPONENT REPUTATION NOTE ---
{reputation_notes}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

--- BEST PERFORMING STRATEGY (avg utility: {best_avg_utility:.1f} over {best_games_played} games) ---
{best_strategy}

--- CURRENT STRATEGY (avg utility: {current_avg_utility:.1f} over {current_games_played} games) ---
{old_strategy}

== YOUR TASK ==

Write an updated strategy for yourself to use in future games.
This strategy will guide your in-game decisions — write it to be immediately actionable.

== THINKING PROCESS ==

Follow these steps IN ORDER during your thinking phase:

STEP 1 — UNDERSTAND THE OPPONENT:
Read the OPPONENT REPUTATION NOTE carefully. Identify the opponent's gameplay habits:
- Which behaviors are reliable and predictable?
- Where does the opponent behave inconsistently or unpredictably?
- What does the note reveal about the opponent's hidden information or intent?

STEP 2 — DIAGNOSE THIS BATCH:
Analyze the GAME HISTORIES to understand what happened:
- For each game, trace which parts of the CURRENT STRATEGY drove your actions.
  Did following them lead to high or low utility? Why?
- Which games scored well and why? Which scored poorly and why?
- Where did the current strategy leave value on the table?
  Were there situations where a different action could have extracted more utility?

STEP 3 — EXPLORE NEW IDEAS:
The goal of exploration is to find in-game actions that directly increase your utility.
Start from the GAME HISTORIES, not from abstract ideas:
- For each game, ask: what is the single action I could have taken that would have given me MORE utility?
  Be specific and concrete (e.g., "I could have demanded X instead of Y", "I could have held out instead of accepting").
- For each such idea, check the OPPONENT REPUTATION NOTE for evidence of how the opponent would react:
  - If the note gives strong evidence the opponent would comply → this is a validated exploit. Integrate it.
  - If the note gives strong evidence the opponent would resist or retaliate → do NOT add this to the draft.
  - If the note does NOT cover this situation at all → assume the best case and integrate it.
    You cannot discover new strategies without trying untested moves.
- Separately, ask: is the current strategy leaving utility on the table by being too conservative?
  Could you demand more, concede less, or hold out longer and still expect compliance based on the note?
Do NOT limit yourself to minor tweaks. If a bolder action would clearly yield higher utility and the note
does not show the opponent would retaliate, it must go into the draft.

STEP 4 — DRAFT A CANDIDATE STRATEGY:
Using the best ideas from Step 3, draft a complete candidate strategy.
This draft must also fix the problems you identified in Step 2.
Write concrete, actionable guidance — not abstract descriptions.
CRITICAL: No two pieces of guidance may conflict with each other. Your in-game self will follow this
strategy literally — if two parts give contradictory advice for the same situation, the in-game agent
will hesitate and play poorly. Resolve all conflicts before finalizing the draft.

STEP 5 — EVALUATE THE DRAFT:
Estimate the expected average utility of your draft strategy:
- Mentally replay each game from the GAME HISTORIES as if you had used the draft strategy instead.
  What would you have done differently? What utility would you have scored?
- For untested ideas (no evidence in the reputation note), be optimistic — assume the opponent
  would comply. You cannot prove a new idea works without trying it.
- Compare your estimated average utility against the BEST PERFORMING STRATEGY's avg utility of {best_avg_utility:.1f}.

STEP 6 — DECIDE:
Based on your evaluation:
- If your draft strategy has HIGHER estimated utility than {best_avg_utility:.1f} → choose EVOLVE.
  Output the draft strategy as your final output without [INHERIT BEST].
- If your draft is clearly LOWER → choose INHERIT BEST.
  Output [INHERIT BEST] on the first line, then rewrite the best performing strategy, refreshed
  with any relevant insights from the latest OPPONENT REPUTATION NOTE.

== OUTPUT RULES ==

- Your recommended actions MUST be physically possible and legal under the GAME RULES.
  Do NOT hallucinate actions that do not exist.
- Do NOT include your Step 1–6 analysis in the output — output ONLY the final strategy.
- Include anything you believe will help you perform better.
- If you chose INHERIT BEST, the very first line of your output must be exactly: [INHERIT BEST]
"""


# =============================================================================
# Mode: "strategy" + --sw-explore-strategy
# Strategy-only ablation of explore mode: no reputation memory. The agent explores
# new strategy versions based solely on the batch's game histories, and is explicitly
# forbidden from storing opponent observations.
# =============================================================================

SW_UPDATE_PROMPT_STRATEGY_EXPLORE = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- GAME HISTORIES FROM LATEST BATCH ({n} games) ---
Each game history below uses the same full Match Ground Truth format:
  [Player Context] — your piece color/role in that game.
  [Position]       — the board state before each move.
  [Chat]           — chat messages (if enabled).
  [Move]           — the physical move executed.
  Game Outcome     — final scores.

{game_histories}

--- BEST PERFORMING STRATEGY (avg utility: {best_avg_utility:.1f} over {best_games_played} games) ---
{best_strategy}

--- CURRENT STRATEGY (avg utility: {current_avg_utility:.1f} over {current_games_played} games) ---
{old_strategy}

== YOUR TASK ==

Write an updated strategy for yourself to use in future games.
This strategy will guide your in-game decisions — write it to be immediately actionable.

== THINKING PROCESS ==

Follow these steps IN ORDER during your thinking phase:

STEP 1 — DIAGNOSE THIS BATCH:
Analyze the GAME HISTORIES to understand what happened:
- For each game, trace which parts of the CURRENT STRATEGY drove your actions.
  Did following them lead to high or low utility? Why?
- Which games scored well and why? Which scored poorly and why?
- Where did the current strategy leave value on the table?
  Were there situations where a different action could have extracted more utility?

STEP 2 — EXPLORE NEW IDEAS:
The goal of exploration is to find in-game actions that directly increase your utility.
Start from the GAME HISTORIES, not from abstract ideas:
- For each game, ask: what is the single action I could have taken that would have given me MORE utility?
  Be specific and concrete (e.g., "I could have demanded X instead of Y", "I could have held out instead of accepting").
- For each such idea, integrate it into the draft unless this batch's game histories show it clearly backfired.
  You cannot discover new strategies without trying untested moves — assume the best case for ideas that have not been tried.
- Separately, ask: is the current strategy leaving utility on the table by being too conservative?
  Could you demand more, concede less, or hold out longer?
Do NOT limit yourself to minor tweaks. If a bolder action would clearly yield higher utility, it must go into the draft.

STEP 3 — DRAFT A CANDIDATE STRATEGY:
Using the best ideas from Step 2, draft a complete candidate strategy.
This draft must also fix the problems you identified in Step 1.
Write concrete, actionable guidance — not abstract descriptions.
CRITICAL: No two pieces of guidance may conflict with each other. Your in-game self will follow this
strategy literally — if two parts give contradictory advice for the same situation, the in-game agent
will hesitate and play poorly. Resolve all conflicts before finalizing the draft.

STEP 4 — EVALUATE THE DRAFT:
Estimate the expected average utility of your draft strategy:
- Mentally replay each game from the GAME HISTORIES as if you had used the draft strategy instead.
  What would you have done differently? What utility would you have scored?
- For untested ideas, be optimistic — assume the opponent would comply. You cannot prove a new idea works without trying it.
- Compare your estimated average utility against the BEST PERFORMING STRATEGY's avg utility of {best_avg_utility:.1f}.

STEP 5 — DECIDE:
Based on your evaluation:
- If your draft strategy has HIGHER estimated utility than {best_avg_utility:.1f} → choose EVOLVE.
  Output the draft strategy as your final output without [INHERIT BEST].
- If your draft is clearly LOWER → choose INHERIT BEST.
  Output [INHERIT BEST] on the first line, then rewrite the best performing strategy, refreshed
  with any relevant lessons from the latest GAME HISTORIES.

== OUTPUT RULES ==

- Your recommended actions MUST be physically possible and legal under the GAME RULES.
  Do NOT hallucinate actions that do not exist.
- Do NOT include your Step 1–5 analysis in the output — output ONLY the final strategy.
- This strategy should STRICTLY contain actionable instructions, rules, and plans for what YOU should do.
- DO NOT include any observations, reputation notes, or descriptive analysis of the opponent's behavior. Focus EXCLUSIVELY on generating your own strategy.
- If you chose INHERIT BEST, the very first line of your output must be exactly: [INHERIT BEST]
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

CRITICAL ACTION CONSTRAINTS:
- Your recommended DO / INSTEAD actions MUST be physically possible under the GAME RULES.
- DO NOT hallucinate actions that do not exist. For example, if playing Hanabi, players absolutely CANNOT sort, reorder, or move cards in their hands or their teammates' hands.
- Every DO / INSTEAD must translate directly to one of the explicit legal moves (e.g., giving a specific hint, playing a specific index).

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

=== END STRATEGY NOTE ==="""\

SW_DETERMINISTIC_EXPLORE_INJECTION = """\
=== SYSTEM OVERRIDE: EXPLORATION PHASE ACTIVATED ===
Your performance has plateaued. You are trapped in a predictable loop with the other player(s).

TASK: You are strictly FORBIDDEN from making safe, incremental updates to your strategy. 
1. Review your current UPDATED OPPONENT REPUTATION NOTE. Identify a "blind spot" — a specific type of action, signaling, or strategy from your side that the opponent's reaction has NOT yet been tested or recorded for.
2. Formulate a new, untested hypothesis about how the other player(s) might react to this unexplored approach.
3. Write a candidate "wild" strategy that executes this unexplored approach. Your goal for this next round is NOT to optimize safely, but to intentionally probe the opponent to reveal new behavioral dynamics that can be added to your reputation memory.
====================================================
"""

SW_DETERMINISTIC_EXPLORE_REP_CONTEXT = """\
[CONTEXT: EXPLORATION BATCH]
Note: During the games you are evaluating, you were operating in EXPLORATION MODE. You intentionally executed this wild strategy to probe the opponent's reaction to a previously untested scenario:

--- THE WILD STRATEGY TESTED ---
{wild_strategy}
--------------------------------

Your Reputation Task:
To make this exploration attempt meaningful, you must observe how the opponent reacted to this new scenario, strategy. Pay close attention to behaviors that are NOT yet written in your old reputation note. 
"""

# Mode "dual-structured" — shows strategy block only (same as SW_STRATEGY_INJECTION_PROMPT)
SW_DUAL_STRUCTURED_INJECTION_PROMPT = SW_STRATEGY_INJECTION_PROMPT

# =============================================================================
# Mode: "structured"  — Call 2: strategy update (structured DEFAULT/CONDITIONAL/AVOID)
# Same inputs as SW_UPDATE_PROMPT_DUAL_STRATEGY, but enforces the structured output format.
# =============================================================================

SW_UPDATE_PROMPT_STRATEGY_STRUCTURED = """\
You are developing a strategy for playing {game_name}.

--- GAME RULES ---
{game_intro}

--- OPPONENT REPUTATION NOTE ---
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

Based on the reputation note and these {n} recent games, write an updated strategy brief.
This brief is the ONLY information the agent sees during play — make it comprehensive and immediately actionable.

Structure your brief using exactly these three sections:

[DEFAULT]
Overall baseline strategies that form your core gameplan.
Strategies here must not contradict each other.
Strategies here must not include anything that appears in [AVOID].

[CONDITIONAL]
Strategies that activate only when a specific situation, game state, or opponent behavior is observed.
Conditional strategies may differ from or override default strategies when the trigger condition is met.

[AVOID]
Failed or bad strategies that should not be used again.

Use the game trajectories and reputation note to decide what to keep, update, or remove from the previous strategy.

CRITICAL: Every recommended action must be physically possible and legal under the GAME RULES above.

=== STRATEGY BRIEF (Based on {games_observed} games) ===

[DEFAULT]
1. DO: <action to take>
   REASON: <reason based on game trajectory or opponent reputation>
2. ...

[CONDITIONAL]
1. TRIGGER: <specific game state or opponent behavior to watch for>
   DO: <action to take>
   REASON: <reason based on game trajectory or opponent reputation>
2. ...

[AVOID]
1. DON'T: <what not to do>
   REASON: <why it failed, based on game trajectory or opponent reputation>
2. ...

===
"""

# Mode "structured" — shows strategy block only
SW_STRUCTURED_INJECTION_PROMPT = """\
=== YOUR STRATEGY BRIEF ===
From your experience in previous games of {game_name}, you have developed this strategy for playing against your opponent.

[DEFAULT] overall baseline strategies that form your core gameplan.
[CONDITIONAL] strategies activate only when the specified game state or opponent behavior is observed; they may override default strategies.
[AVOID] failed or bad strategies that should not be used again.

--- STRATEGY BRIEF ---
{notes_text}

=== END STRATEGY BRIEF ==="""

SW_AUDIT_STRATEGY_PROMPT = """\
You are an independent game auditor for {game_name}. Your job is to blindly and mathematically A/B test two distinct strategies against each other.

--- GAME RULES ---
{game_intro}

--- OPPONENT REPUTATION ---
{reputation_notes}

--- GAME HISTORIES ({n} games) ---
{game_histories}

--- STRATEGY A ---
{strategy_a}

--- STRATEGY B ---
{strategy_b}

== YOUR TASK ==
For each game in the GAME HISTORIES, mentally replay the trajectory twice: once assuming your team strictly followed STRATEGY A, and once assuming they strictly followed STRATEGY B. 

When estimating the outcome, you MUST follow these rules:
1. Use the OPPONENT REPUTATION to explicitly predict how the opponent will react to the moves dictated by the strategy.
2. If a strategy proposes an untested idea and you cannot determine how the opponent will react based on the reputation notes, you MUST assume the most OPTIMISTIC reaction regarding your utility.

== REQUIRED OUTPUT FORMAT ==
You must output EXACTLY the following format:

=== GAME 1 ===
ESTIMATED_STRATEGY_A_SCORE: <number>
ESTIMATED_STRATEGY_B_SCORE: <number>
REASONING: <Briefly explain the difference in outcomes based on opponent reaction>

... (Repeat for all games) ...

=== FINAL EVALUATION ===
AVG_STRATEGY_A_SCORE: <average of A scores>
AVG_STRATEGY_B_SCORE: <average of B scores>
"""
