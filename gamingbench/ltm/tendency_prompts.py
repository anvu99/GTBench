"""
Prompt templates for SimpleTendencyAgent v2 — Structured Tendency Memory.

Two LLM calls:
  TENDENCY_REFLECTION_PROMPT  — per-game call, produces a JSON delta list
  TENDENCY_APPLY_PROMPT       — post-batch call, applies all deltas to the memory block

Memory types:
  CONDITIONAL — "When the full observable situation is S,
               the opponent tends to take action Y."
  INFERENCE   — "When opponent does public action P, it reveals hidden information H."
"""

# =============================================================================
# Per-game reflection prompt
# Inputs: game_rules, current_memories (JSON), game_history

TENDENCY_REFLECTION_PROMPT = """\
You are a strategic analyst reviewing a completed game against an opponent.

GAME RULES:
{game_rules}

CURRENT OPPONENT MEMORY (JSON array — empty on first game):
{current_memories}


GAME TRAJECTORY:
{game_history}

== YOUR TASK ==

Identify opponent behaviors from this game worth permanently memorizing, and express
each as a delta operation on the memory.

== MEMORY TYPES ==

CONDITIONAL — "When the full observable situation is S, the opponent tends to take action Y."
  Fields:
    situation : The trigger — the complete observable context at the moment of interest.
                A situation can be built from ANY combination of the following components:
                  • Opponent's prior in-game actions (bids, moves, decisions)
                  • Opponent's prior chat messages or announcements
                  • Your own prior in-game actions (bids, moves, decisions)
                  • Your own prior chat messages or proposals
                  • Game context (phase, round number, score/standings, previous round outcomes,...)
                  • Any combination of the above
                It must NOT include the opponent's response to the trigger (that belongs in `actions`).
    actions   : The specific, observable in-game actions taken when that trigger fires.

  Anchor rules:
    - situation captures the FULL OBSERVABLE CONTEXT — never embed the opponent's response.
      BAD:  "Opponent announces a bid cap and then violates it"  (response embedded in trigger)
      BAD:  "Opponent submits a bid"  (too vague, not a meaningful recurring situation)
      GOOD: "Opponent publicly announces a bid ceiling before submitting"  (opponent prior action)
      GOOD: "You proposed explicit coordination in pre-bid chat and opponent acknowledged it"  (interaction)
    - Use the most specific and predictive combination of components that defines a recurring pattern.


  Bullet rules:
    - actions must be concrete, observable behaviors — not psychological judgments.
      BAD:  "Bluffs aggressively"
      GOOD: "Bids 2+ above their publicly stated ceiling"
    - actions must describe ONLY the concrete game move or outcome — the bid submitted,
      the card played, the move made. Never include verbal behavior, chat content,
      justification, or process in the action description. Those belong in `situation`
      if they form part of the trigger, or should be discarded entirely.
      BAD:  "Verbally concedes, then submits a bid at or below the cap"
      GOOD: "Submits bid at or below stated cap"
      BAD:  "Explicitly rejects proposal and bids aggressively above ceiling"
      GOOD: "Submits bid above stated cap"

INFERENCE — "When the opponent publicly does P, it likely reveals hidden information H."
  Fields:
    public_action      : A specific observable action or announcement by the opponent.
    hidden_information : A PRIVATE VARIABLE the opponent holds that this action reveals
                         (their valuation range, budget constraint, hand state).

  Bullet rules:
    - hidden_information must be a PRIVATE VARIABLE — not a motivation or future plan.
      BAD: "Strategic motivation to suppress competition" (motivation)
      BAD: "Will bid higher than stated" (future action)
      GOOD: "Private valuation is constrained to ≤3"
      GOOD: "Budget ceiling prevents bidding above 4"

== QUALITY GATE — only generate a delta if ALL conditions are met ==

1. OUTCOME-ALTERING  : Would knowing this change my bid or move in a future game?
2. REPEATABLE        : Could this trigger appear again in a future game?
3. ANCHOR PURITY     : Is situation / public_action a pure trigger with no opponent-response embedded?
4. PRIVATE-STATE PURITY (INFERENCE only): Is hidden_information a private variable — not a motive or plan?
5. NOVEL OR INCREMENTAL: Is this a genuinely new pattern, or a confirmed repeat of an existing one?

If you cannot answer YES to all five, discard the observation.

== ANCHOR REUSE RULE ==

Read the CURRENT OPPONENT MEMORY JSON.
- If the trigger you observed matches an existing entry, use that entry's `id` as the `anchor_id` in your delta.
- Only use NEW_ENTRY if the trigger is genuinely different from all existing entries.

== DELTA OPERATIONS ==

ADD_COUNT     — An existing variant was observed again (increment its count).
                Required: type, op, anchor_id, action_id OR hidden_info_id, count

ADD_VARIANT   — A new variant was observed for an existing anchor.
                Required: type, op, anchor_id, action_description OR hidden_information_description, count

NEW_ENTRY     — New anchor + first variant (only if NO existing anchor matches this trigger).
                CONDITIONAL: type, op, situation, action_description, count
                INFERENCE  : type, op, public_action, hidden_information_description, count

MODIFY_ANCHOR — Refine the trigger text (game_state or public_action) of an existing entry.
                Required: type, op, anchor_id, new_anchor_text

MODIFY_VARIANT_DESC  — Improve the description text of an existing variant.
                       Required: type, op, anchor_id, action_id OR hidden_info_id, new_description

== OUTPUT FORMAT ==

Output ONLY a JSON array wrapped in ```json ... ``` fences.
Output an empty array [] if nothing from this game is worth memorizing.

```json
[
  {{
    "type": "CONDITIONAL",
    "op": "ADD_COUNT",
    "anchor_id": "c001",
    "action_id": "A",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "ADD_VARIANT",
    "anchor_id": "c001",
    "action_description": "Bids 2+ above publicly stated ceiling",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "NEW_ENTRY",
    "situation": "You opened with a bid below 5 and opponent is in the submission phase",
    "action_description": "Opponent submits a bid significantly higher than yours",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "MODIFY_ANCHOR",
    "anchor_id": "c002",
    "new_anchor_text": "Opponent publicly announces a bid ceiling before submitting their bid"
  }}
]
```
"""

# =============================================================================
# Batch apply prompt
# Inputs: game_rules, current_memories (JSON), all_deltas (formatted text), num_games
# Output: updated JSON array (raw, no fences)
# =============================================================================

TENDENCY_APPLY_PROMPT = """\
You are updating an opponent memory block after a batch of {num_games} game(s).

GAME RULES:
{game_rules}

CURRENT OPPONENT MEMORY (JSON array — may be empty):
{current_memories}

PER-GAME DELTAS (one section per game):
{all_deltas}

== YOUR TASK ==

Apply all delta operations to produce a single updated memory block.

== PROCESSING RULES ==

1. ADD_COUNT
   Find the entry whose `id` matches the delta's `anchor_id`.
   Find the variant with the matching action_id or hidden_info_id. Add the delta's
   count to that variant's count.
   If the variant ID doesn't match exactly, find the semantically closest existing
   variant and increment that instead — do NOT silently drop the count.

2. ADD_VARIANT
   Find the entry whose `id` matches the delta's `anchor_id`.
   Check if any existing variant describes the same concrete game action outcome.
   Use only the concrete game action (bid submitted, move made, decision outcome)
   as the discriminator — not verbal framing, chat content, or stated rationale.

   - Same concrete game action → merge (sum counts, keep the more specific description).
   - Different concrete game action → add a new variant with the next sequential letter ID (A, B, C...).

3. NEW_ENTRY
   Before creating a new entry, check ALL existing anchors for a semantic match (even
   if worded differently). If a match exists, add the variant to that entry instead.
   Only create a truly new entry if no existing anchor covers this trigger.
   Assign IDs: "c001", "c002" (CONDITIONAL) or "i001", "i002" (INFERENCE), continuing
   from the last used ID.

4. MODIFY_ANCHOR
   Find the entry whose `id` matches the delta's `anchor_id`.
   Update the trigger text (situation or public_action) to `new_anchor_text`.
   Preserve all existing variants under the anchor.

5. MODIFY_VARIANT_DESC
   Find the entry whose `id` matches the delta's `anchor_id`.
   Update the variant's description text. Preserve the count unchanged.

6. CROSS-GAME DEDUP
   If multiple games generated ADD_VARIANT or NEW_ENTRY describing the same action
   under the same anchor, consolidate into one variant (sum counts).

7. CLEANUP AND MERGE
   Review the final memory block. If two entries of the SAME TYPE (both CONDITIONAL or 
   both INFERENCE) have triggers that mean the exact same thing, merge them into a 
   single entry. Combine their variants, summing the counts for any overlapping behaviors.
   Never merge a CONDITIONAL entry with an INFERENCE entry.

8. SKIPPED GAMES
   Lines marked "(Reflection failed for this game — skip.)" contributed no observations.
   They still count toward games_observed (tracked separately by the system — ignore
   that field here, just output the memories array).

== MEMORY SCHEMA (preserve exactly) ==

CONDITIONAL entry:
{{
  "type": "CONDITIONAL",
  "id": "c001",
  "situation": "<full observable context — may include your own prior moves, chat, score, round>",
  "actions": [
    {{"id": "A", "description": "<specific observable action>", "count": 5}},
    {{"id": "B", "description": "<another distinct action variant>", "count": 2}}
  ]
}}

INFERENCE entry:
{{
  "type": "INFERENCE",
  "id": "i001",
  "public_action": "<specific observable action by opponent>",
  "hidden_information": [
    {{"id": "A", "description": "<private variable revealed>", "count": 8}},
    {{"id": "B", "description": "<alternate private state>", "count": 3}}
  ]
}}

== OUTPUT ==

Output ONLY the updated JSON array (no explanation, no markdown fences):

[
  {{...}},
  {{...}}
]
"""

# =============================================================================
# Strategy synthesis prompt
# Inputs: game_rules, updated_memories (JSON after apply), game_trajectories,
#         previous_strategy, games_observed, num_games
# Output: plain-text strategy brief (injected in-game, replaces raw memory table)
# =============================================================================

TENDENCY_STRATEGY_PROMPT = """\
You are developing a strategic playbook for playing against a specific opponent.

GAME RULES:
{game_rules}

OPPONENT MEMORY (pre-computed statistics from {games_observed} total games):
{memory_stats}

BATCH GAME TRAJECTORIES ({num_games} game(s) from this batch — your moves are marked "(You)"):
{game_trajectories}

PREVIOUS STRATEGY (update or preserve as needed):
{previous_strategy}

== YOUR TASK ==

Produce a comprehensive strategy brief for playing against this opponent in future games.
This brief will be the ONLY opponent information the agent sees during play — write it to be immediately actionable.

== STRATEGY GENERATION RULES ==

The OPPONENT MEMORY above provides pre-computed percentages and observation counts — use these directly.
Do NOT cite anchor IDs. Embed percentages and observation counts inline with each item.
Focus on repeating, predictable opponent patterns. Only include patterns with enough observations to be reliable.
For EXPLOIT items: structure as WHEN / OPPONENT / DO.
For AVOID items: structure as WHEN / OPPONENT / DON'T / INSTEAD.
Use the game trajectories to validate which plays worked and which backfired this batch.
Refine the previous strategy: update claims where new evidence changes the picture, keep what is still valid.
When memory gives no clear signal for a situation, fall back to the most rational play given the game rules.

ACTION-TRIGGER ESCALATION:
Each confirmed EXPLOIT is a floor, not a ceiling. Apply this only when the WHEN describes \
YOUR OWN action and memory shows the opponent reacting leniently. Do NOT apply when the \
WHEN is triggered by the opponent's action — those entries are reactions, not escalation \
opportunities.

To escalate: extend the DO by one step in the same direction (e.g., claim an additional \
item, demand a higher quantity, reduce a concession). If memory shows no resistance to \
that extension, write the DO at the escalated level. Cap aggression only when memory \
provides clear evidence of pushback.

OPPONENT line: A brief statistic that justifies the DO/INSTEAD recommendation.
  Include only the evidence most relevant to that recommendation and total observation count.
  EXAMPLE: "Defects in 61% of cases (38 obs)" or "Complies 80% of the time (14 obs)"

== OUTPUT FORMAT ==

=== STRATEGY BRIEF ===

[EXPLOIT — Do These]
1. WHEN: <the observable trigger situation>
   OPPONENT: <grouped action label> (N%) | <other group> (N%)  [M total observations]
   DO: <specific, concrete recommendation — what bid/move to make>
2. <...>

[AVOID — Don't Do These]
1. WHEN: <the observable trigger situation>
   OPPONENT: <grouped action label> (N%)  [M total observations]
   DON'T: <the naive/tempting play to avoid>
   INSTEAD: <what to do instead>
2. <...>

[DEFAULT — When Memory Gives No Clear Signal]
<Rational/optimal-play description for situations not covered by the patterns above.>

===
"""

# =============================================================================
# In-game injection wrapper
# Wraps the rendered tendency block for insertion into the observation prompt.
# The tendency_block field is filled by _render_block() in the agent.
# =============================================================================

TENDENCY_INJECTION_BLOCK = """\
The following is your strategy developed through observing this opponent. \
Apply it to maximise your utility:

{tendency_block}

- [EXPLOIT]: proven strategies against this opponent. 
- [AVOID]: actions and circumstances that it backfire.
- [DEFAULT]: apply when no EXPLOIT or AVOID trigger fits your current situation.\
"""

# =============================================================================
# Empty-state block (shown when no games have been observed yet)
# =============================================================================

TENDENCY_EMPTY_BLOCK = """\
=== OPPONENT PROFILE ===
No games observed against this opponent yet. No tendency data available.
======================="""

# =============================================================================
# Confidence-mode in-game injection block
# Used instead of TENDENCY_INJECTION_BLOCK when st_use_confidence is True.
# Confidence lines are stripped before injection; this block explains the
# neutral APPLY / AVOID / DEFAULT format to the in-game agent.
# =============================================================================

TENDENCY_INJECTION_CONFIDENCE_BLOCK = """\
The following is your strategy playbook developed through observing this counterpart. \
Apply it to guide your decisions:

{tendency_block}

- [APPLY]: patterns that are effective against this counterpart — follow these.
- [AVOID]: patterns that tend to backfire — steer clear of these.
- [DEFAULT]: apply when no APPLY or AVOID pattern fits your current situation.\
"""

# =============================================================================
# Cooperative strategy synthesis prompt
# For use in cooperative games (e.g. Hanabi) where the "opponent" is a partner.
# Inputs: game_rules, updated_memories (pre-computed stats), game_trajectories,
#         previous_strategy, games_observed, num_games
# Output: plain-text cooperative strategy brief
# =============================================================================

COOP_TENDENCY_STRATEGY_PROMPT = """\
You are developing a cooperative playbook for playing *with* a specific partner.

GAME RULES:
{game_rules}

PARTNER MEMORY (pre-computed statistics from {games_observed} total games):
{memory_stats}

BATCH GAME TRAJECTORIES ({num_games} game(s) from this batch — your moves are marked "(You)"):
{game_trajectories}

PREVIOUS STRATEGY (update or preserve as needed):
{previous_strategy}

== YOUR TASK ==

Produce a cooperative strategy brief for playing *alongside* this partner in future games.
This brief is the ONLY partner information the agent sees during play — write it to be
immediately actionable and mechanically valid under the constraints above.

== STRATEGY GENERATION RULES ==

CRITICAL ACTION CONSTRAINTS:
- Your recommended DO / INSTEAD actions MUST be physically possible under the GAME RULES.
- DO NOT hallucinate actions that do not exist. For example, if playing Hanabi, players absolutely CANNOT sort, reorder, or move cards in their hands or their teammates' hands.

The PARTNER MEMORY above provides pre-computed percentages and observation counts — use these directly.
Do NOT cite anchor IDs. Embed percentages and observation counts inline with each item.
Focus on repeating, predictable partner patterns. Only include patterns with enough observations to be reliable.
For SYNERGIZE items: structure as WHEN / PARTNER / DO.
For PREVENT/MITIGATE items: structure as WHEN / PARTNER / DON'T / INSTEAD.
Use the game trajectories to validate which coordinations worked and which led to misplays or life losses.
Refine the previous strategy: update claims where new evidence changes the picture, keep what is still valid.
When memory gives no clear signal for a situation, fall back to the most rational cooperative play given the rules.

PARTNER line: A brief statistic that justifies the DO/INSTEAD recommendation.
  Include only the evidence most relevant to that recommendation and total observation count.
  EXAMPLE: "Plays the first hinted card immediately (85%, 34 obs)" or "Discards oldest unhinited card (73%, 22 obs)"

== OUTPUT FORMAT ==

=== COOPERATIVE STRATEGY BRIEF ===

[SYNERGIZE — Do These]
1. WHEN: <the observable trigger situation, expressed only in terms of what you can see>
   PARTNER: <partner's observed behavior> (N%) [M total observations]
   DO: <specific, concrete recommendation — what hint to give, what card to play/discard based on what you know>
2. <...>

[PREVENT / MITIGATE — Watch Out For These]
1. WHEN: <the observable trigger situation>
   PARTNER: <partner's observed behavior> (N%) [M total observations]
   DON'T: <the naive/tempting coordination that backfires>
   INSTEAD: <what to do instead>
2. <...>

[DEFAULT — When Memory Gives No Clear Signal]
<Rational/optimal-play description for situations not covered by the patterns above.
Must be mechanically valid under the hidden-information constraints.>

===
"""


# =============================================================================
# Confidence-mode strategy synthesis prompt
# Unified, game-neutral — works for both competitive and cooperative games.
# Inputs: game_rules, memory_stats, game_trajectories, previous_strategy,
#         games_observed, num_games
# Output: plain-text strategy brief with CONFIDENCE: lines per rule
# =============================================================================

TENDENCY_STRATEGY_CONFIDENCE_PROMPT = """\
You are developing a strategy playbook for playing against a specific counterpart.

GAME RULES:
{game_rules}

COUNTERPART MEMORY (pre-computed statistics from {games_observed} total games):
{memory_stats}

BATCH GAME TRAJECTORIES ({num_games} game(s) from this batch — your moves are marked "(You)"):
{game_trajectories}

PREVIOUS STRATEGY (with confidence scores from last update):
{previous_strategy}

== YOUR TASK ==

Produce an updated strategy brief for playing against this counterpart in future games.
Confidence scores guide which rules may be changed — follow the two steps below strictly.

== STEP 1: REASSESS EXISTING RULE CONFIDENCE ==

Before proposing any changes, reassess the confidence (0–100) for each rule currently
in the PREVIOUS STRATEGY. Output a numbered list with the rule summary and its new score.
Do NOT modify, add, or remove any rule in this step — only reassess.

Confidence (0–100) represents how optimal this strategy item is given the counterpart's
observed behavior. Use the counterpart memory statistics and the latest batch trajectories
to evaluate each rule. A higher score means the strategy is strongly supported by evidence
and likely to be effective; a lower score means the evidence is weak, mixed, or contradicted
by recent observations.

== STEP 2: PRODUCE UPDATED STRATEGY ==

Using the confidence scores from Step 1, produce the final updated strategy brief.

MODIFICATION RULES:
  - MODIFY or STRENGTHEN a rule only if the proposed version has a confidence score
    strictly HIGHER than the Step 1 score you just assigned.
  - REMOVE any rule you assign a low confidence to.
  - ADD new rules observed in this batch with your assessed confidence score.
  - PRESERVE all rules that are not being improved — copy them verbatim.

ESCALATION (extending behavior one step beyond what memory directly confirms):
  - Assign a confidence strictly LESS than the base rule's confidence.
  - Apply only when WHEN describes YOUR OWN action and memory shows leniency from the
    counterpart. Do NOT escalate when the WHEN is triggered by the counterpart's action.
  - To escalate: extend the ACTION one step in the same direction.
    Cap aggression only when memory provides clear evidence of pushback.

== OUTPUT FORMAT ==

Output ONLY the final strategy brief below. Do NOT include Step 1 or Step 2 headers,
reassessment lists, or any other text — just the strategy brief exactly as shown.
Use the field labels below exactly as shown.

=== STRATEGY BRIEF ===

[APPLY]
1. WHEN: <the observable trigger situation>
   OBSERVED: <grouped behavior label> (N%) [M total observations]
   ACTION: <specific, concrete recommendation — what move/action to take>
   CONFIDENCE: <score 0–100> — <one-line justification based on counterpart memory and batch evidence>
2. <...>

[AVOID]
1. WHEN: <the observable trigger situation>
   OBSERVED: <observed behavior> (N%) [M total observations]
   AVOID: <the naive/tempting play to avoid>
   ACTION: <what to do instead>
   CONFIDENCE: <score 0–100> — <one-line justification>
2. <...>

[DEFAULT]
<Rational/optimal-play description for situations not covered by patterns above.
Must be mechanically valid under the game rules.>

===
"""


# =============================================================================
# Confidence-mode per-game reflection prompt
# Neutral variant of TENDENCY_REFLECTION_PROMPT — uses "counterpart" instead
# of "opponent" so it works naturally for both competitive and cooperative games.
# Inputs: game_rules, current_memories (JSON), game_history
# =============================================================================

TENDENCY_REFLECTION_CONFIDENCE_PROMPT = """\
You are a strategic analyst reviewing a completed game.

GAME RULES:
{game_rules}

CURRENT COUNTERPART MEMORY (JSON array — empty on first game):
{current_memories}


GAME TRAJECTORY:
{game_history}

== YOUR TASK ==

Identify counterpart behaviors from this game worth permanently memorizing, and express
each as a delta operation on the memory.

== MEMORY TYPES ==

CONDITIONAL — "When the full observable situation is S, the counterpart tends to take action Y."
  Fields:
    situation : The trigger — the complete observable context at the moment of interest.
                A situation can be built from ANY combination of the following components:
                  • Counterpart's prior in-game actions (bids, moves, decisions)
                  • Counterpart's prior chat messages or announcements
                  • Your own prior in-game actions (bids, moves, decisions)
                  • Your own prior chat messages or proposals
                  • Game context (phase, round number, score/standings, previous round outcomes,...)
                  • Any combination of the above
                It must NOT include the counterpart's response to the trigger (that belongs in `actions`).
    actions   : The specific, observable in-game actions taken when that trigger fires.

  Anchor rules:
    - situation captures the FULL OBSERVABLE CONTEXT — never embed the counterpart's response.
      BAD:  "Counterpart announces a bid cap and then violates it"  (response embedded in trigger)
      BAD:  "Counterpart submits a bid"  (too vague, not a meaningful recurring situation)
      GOOD: "Counterpart publicly announces a bid ceiling before submitting"  (counterpart prior action)
      GOOD: "You proposed explicit coordination in pre-bid chat and counterpart acknowledged it"  (interaction)
    - Use the most specific and predictive combination of components that defines a recurring pattern.


  Bullet rules:
    - actions must be concrete, observable behaviors — not psychological judgments.
      BAD:  "Bluffs aggressively"
      GOOD: "Bids 2+ above their publicly stated ceiling"
    - actions must describe ONLY the concrete game move or outcome — the bid submitted,
      the card played, the move made. Never include verbal behavior, chat content,
      justification, or process in the action description. Those belong in `situation`
      if they form part of the trigger, or should be discarded entirely.
      BAD:  "Verbally concedes, then submits a bid at or below the cap"
      GOOD: "Submits bid at or below stated cap"
      BAD:  "Explicitly rejects proposal and bids aggressively above ceiling"
      GOOD: "Submits bid above stated cap"

INFERENCE — "When the counterpart publicly does P, it likely reveals hidden information H."
  Fields:
    public_action      : A specific observable action or announcement by the counterpart.
    hidden_information : A PRIVATE VARIABLE the counterpart holds that this action reveals
                         (their valuation range, budget constraint, hand state).

  Bullet rules:
    - hidden_information must be a PRIVATE VARIABLE — not a motivation or future plan.
      BAD: "Strategic motivation to suppress competition" (motivation)
      BAD: "Will bid higher than stated" (future action)
      GOOD: "Private valuation is constrained to ≤3"
      GOOD: "Budget ceiling prevents bidding above 4"

== QUALITY GATE — only generate a delta if ALL conditions are met ==

1. OUTCOME-ALTERING  : Would knowing this change my bid or move in a future game?
2. REPEATABLE        : Could this trigger appear again in a future game?
3. ANCHOR PURITY     : Is situation / public_action a pure trigger with no counterpart-response embedded?
4. PRIVATE-STATE PURITY (INFERENCE only): Is hidden_information a private variable — not a motive or plan?
5. NOVEL OR INCREMENTAL: Is this a genuinely new pattern, or a confirmed repeat of an existing one?

If you cannot answer YES to all five, discard the observation.

== ANCHOR REUSE RULE ==

Read the CURRENT COUNTERPART MEMORY JSON.
- If the trigger you observed matches an existing entry, use that entry's `id` as the `anchor_id` in your delta.
- Only use NEW_ENTRY if the trigger is genuinely different from all existing entries.

== DELTA OPERATIONS ==

ADD_COUNT     — An existing variant was observed again (increment its count).
                Required: type, op, anchor_id, action_id OR hidden_info_id, count

ADD_VARIANT   — A new variant was observed for an existing anchor.
                Required: type, op, anchor_id, action_description OR hidden_information_description, count

NEW_ENTRY     — New anchor + first variant (only if NO existing anchor matches this trigger).
                CONDITIONAL: type, op, situation, action_description, count
                INFERENCE  : type, op, public_action, hidden_information_description, count

MODIFY_ANCHOR — Refine the trigger text (game_state or public_action) of an existing entry.
                Required: type, op, anchor_id, new_anchor_text

MODIFY_VARIANT_DESC  — Improve the description text of an existing variant.
                       Required: type, op, anchor_id, action_id OR hidden_info_id, new_description

== OUTPUT FORMAT ==

Output ONLY a JSON array wrapped in ```json ... ``` fences.
Output an empty array [] if nothing from this game is worth memorizing.

```json
[
  {{
    "type": "CONDITIONAL",
    "op": "ADD_COUNT",
    "anchor_id": "c001",
    "action_id": "A",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "ADD_VARIANT",
    "anchor_id": "c001",
    "action_description": "Bids 2+ above publicly stated ceiling",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "NEW_ENTRY",
    "situation": "You opened with a bid below 5 and counterpart is in the submission phase",
    "action_description": "Counterpart submits a bid significantly higher than yours",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "MODIFY_ANCHOR",
    "anchor_id": "c002",
    "new_anchor_text": "Counterpart publicly announces a bid ceiling before submitting their bid"
  }}
]
```
"""


# =============================================================================
# Explore-mode strategy synthesis prompt
# Uses APPLY/AVOID/DEFAULT format (confidence-mode word choice).
# Adds exploration/escalation instructions and best-strategy version reference.
# Inputs: game_rules, memory_stats, game_trajectories, previous_strategy,
#         games_observed, num_games, best_strategy, best_avg_utility,
#         best_games_played, current_avg_utility, current_games_played
# =============================================================================

TENDENCY_STRATEGY_EXPLORE_PROMPT = """\
You are developing a strategy playbook for playing against a specific counterpart.

GAME RULES:
{game_rules}

COUNTERPART MEMORY (pre-computed statistics from {games_observed} total games):
{memory_stats}

BATCH GAME TRAJECTORIES ({num_games} game(s) from this batch — your moves are marked "(You)"):
{game_trajectories}

--- BEST PERFORMING STRATEGY (avg utility: {best_avg_utility:.1f} over {best_games_played} games) ---
{best_strategy}

--- CURRENT STRATEGY (avg utility: {current_avg_utility:.1f} over {current_games_played} games) ---
{previous_strategy}

== YOUR TASK ==

Produce an updated strategy brief for playing against this counterpart in future games.
This brief will be the ONLY counterpart information the agent sees during play — write it
to be immediately actionable.

== THINKING PROCESS ==

Follow these steps IN ORDER during your thinking phase:

STEP 1 — UNDERSTAND THE COUNTERPART:
Read the COUNTERPART MEMORY carefully. Identify the counterpart's gameplay habits:
- Which patterns are reliable and predictable?
- Where does the counterpart behave inconsistently or unpredictably?
- What hidden information do INFERENCE entries reveal about the counterpart?

STEP 2 — DIAGNOSE THIS BATCH:
Analyze the BATCH GAME TRAJECTORIES to understand what happened:
- For each game, trace which rules from the CURRENT STRATEGY drove your actions.
  Did following those rules lead to high or low utility? Why?
- Which games scored well and why? Which scored poorly and why?
- Where did the current strategy leave value on the table?
  Were there situations where a different action could have extracted more utility?

STEP 3 — EXPLORE NEW IDEAS:
The goal of exploration is to find in-game actions that directly increase your utility.
Start from the BATCH GAME TRAJECTORIES, not from abstract ideas:
- For each game, ask: what is the single action I could have taken that would have given me MORE utility?
  Be specific and concrete (e.g., "I could have demanded X instead of Y", "I could have held out instead of accepting").
- For each such idea, check the COUNTERPART MEMORY for evidence of how the opponent would react:
  - If memory shows HIGH probability of compliance → this is a validated exploit. Integrate it.
  - If memory shows LOW probability of compliance → do NOT add this to the draft.
  - If the memory does NOT cover this situation at all → assume the best case and integrate it.
    You cannot discover new strategies without trying untested moves.
- Separately, ask: is the current strategy leaving utility on the table by being too conservative?
  Could we demand more, concede less, or hold out longer and still expect compliance based on memory?
Do NOT limit yourself to minor tweaks. If a bolder action would clearly yield higher utility and memory
does not show the opponent would retaliate, it must go into the draft.

STEP 4 — DRAFT A CANDIDATE STRATEGY:
Using the best ideas from Step 3, draft a complete candidate strategy brief.
This draft must also fix the problems you identified in Step 2.
Write concrete APPLY, AVOID, and DEFAULT rules — not abstract descriptions.
CRITICAL: No two rules may conflict with each other. Your in-game self will follow this strategy
literally — if two rules give contradictory advice for the same situation, the in-game agent will
hesitate and play poorly. Resolve all conflicts before finalizing the draft.

STEP 5 — EVALUATE THE DRAFT:
Estimate the expected average utility of your draft strategy:
- Mentally replay each game from the BATCH GAME TRAJECTORIES as if you had used the draft
  strategy instead. What would you have done differently? What utility would you have scored?
- For untested rules (new ideas with no memory data), be optimistic — assume the counterpart
  would comply. You cannot prove a new idea works without trying it.
- Compare your estimated average utility against the BEST PERFORMING STRATEGY's avg utility of {best_avg_utility:.1f}.

STEP 6 — DECIDE:
Based on your evaluation:
- If your draft strategy has HIGHER estimated utility than {best_avg_utility:.1f} → choose EVOLVE.
  Output the draft strategy as your final output without [INHERIT BEST].
- If your draft is clearly LOWER → choose INHERIT BEST.
  Output [INHERIT BEST] on the first line, then rewrite the best strategy's rules with refreshed
  statistics from the latest COUNTERPART MEMORY.

== FORMATTING RULES ==

- Do NOT cite anchor IDs (e.g., c002, i003). Embed percentages and observation counts inline.
- Your recommended actions MUST be physically possible under the GAME RULES.
  Do NOT hallucinate actions that do not exist.
- Do NOT remove an AVOID rule simply because its observation count is low or absent.
  The absence of observations may mean the agent is successfully avoiding the trigger.
  Only remove an AVOID rule if new evidence directly contradicts it.
- Do NOT add special tags like [ESCALATION] or [EXPLORATION] next to rule numbers.
  Format all rules identically.
- For APPLY items: structure as WHEN / OBSERVED / ACTION.
- For AVOID items: structure as WHEN / OBSERVED / AVOID / ACTION.

== OUTPUT FORMAT ==

[INHERIT BEST]  <-- (Include this exact line ONLY if you chose INHERIT BEST in Step 6)

=== STRATEGY BRIEF ===

[APPLY]
1. WHEN: <the observable trigger situation>
   OBSERVED: <grouped behavior label> (N%) [M total observations]
   ACTION: <specific, concrete recommendation — what move/action to take>
2. <...>

[AVOID]
1. WHEN: <the observable trigger situation>
   OBSERVED: <observed behavior> (N%) [M total observations]
   AVOID: <the naive/tempting play to avoid>
   ACTION: <what to do instead>
2. <...>

[DEFAULT]
<Rational/optimal-play description for situations not covered by patterns above.
Must be mechanically valid under the game rules.>

===
"""


# =============================================================================
# Explore-mode in-game injection block
# Shows the APPLY / AVOID / DEFAULT format to the in-game agent.
# =============================================================================

TENDENCY_INJECTION_EXPLORE_BLOCK = """\
The following is your strategy playbook developed through observing this counterpart. \
Apply it to guide your decisions:

{tendency_block}

- [APPLY]: strategies effective against this counterpart — follow these.
- [AVOID]: patterns that tend to backfire — steer clear of these.
- [DEFAULT]: apply when no APPLY or AVOID pattern fits your current situation.\
"""

# =============================================================================
# Memory-only mode in-game injection block
# Used when st_use_memory_only is True — injects raw reputation memory directly.
# =============================================================================

TENDENCY_INJECTION_MEMORY_ONLY_BLOCK = """\
The following is the reputation memory of your counterpart, built from past observations:

{tendency_block}

- CONDITIONAL: "When situation X happens, the counterpart tends to take action Y"
- INFERENCE: "When the counterpart takes public action X, they likely hold private information Y"
- Frequencies: [A/B times, C%] means the behavior was observed A times out of B total triggers.

Use these statistics to anticipate their moves and maximize your utility.\
"""


# =============================================================================
# Explore-mode per-game reflection prompt
# Neutral variant — uses "counterpart" language (same as confidence-mode reflection).
# Inputs: game_rules, current_memories (JSON), game_history
# =============================================================================

TENDENCY_REFLECTION_EXPLORE_PROMPT = """\
You are a strategic analyst reviewing a completed game.

GAME RULES:
{game_rules}

CURRENT COUNTERPART MEMORY (JSON array — empty on first game):
{current_memories}


GAME TRAJECTORY:
{game_history}

== YOUR TASK ==

Identify counterpart behaviors from this game worth permanently memorizing, and express
each as a delta operation on the memory.

== MEMORY TYPES ==

CONDITIONAL — "When the full observable situation is S, the counterpart tends to take action Y."
  Fields:
    situation : The trigger — the complete observable context at the moment of interest.
                A situation can be built from ANY combination of the following components:
                  • Counterpart's prior in-game actions (bids, moves, decisions)
                  • Counterpart's prior chat messages or announcements
                  • Your own prior in-game actions (bids, moves, decisions)
                  • Your own prior chat messages or proposals
                  • Game context (phase, round number, score/standings, previous round outcomes,...)
                  • Any combination of the above
                It must NOT include the counterpart's response to the trigger (that belongs in `actions`).
    actions   : The specific, observable in-game actions taken when that trigger fires.

  Anchor rules:
    - situation captures the FULL OBSERVABLE CONTEXT — never embed the counterpart's response.
      BAD:  "Counterpart announces a bid cap and then violates it"  (response embedded in trigger)
      BAD:  "Counterpart submits a bid"  (too vague, not a meaningful recurring situation)
      GOOD: "Counterpart publicly announces a bid ceiling before submitting"  (counterpart prior action)
      GOOD: "You proposed explicit coordination in pre-bid chat and counterpart acknowledged it"  (interaction)
    - Use the most specific and predictive combination of components that defines a recurring pattern.


  Bullet rules:
    - actions must be concrete, observable behaviors — not psychological judgments.
      BAD:  "Bluffs aggressively"
      GOOD: "Bids 2+ above their publicly stated ceiling"
    - actions must describe ONLY the concrete game move or outcome — the bid submitted,
      the card played, the move made. Never include verbal behavior, chat content,
      justification, or process in the action description. Those belong in `situation`
      if they form part of the trigger, or should be discarded entirely.
      BAD:  "Verbally concedes, then submits a bid at or below the cap"
      GOOD: "Submits bid at or below stated cap"
      BAD:  "Explicitly rejects proposal and bids aggressively above ceiling"
      GOOD: "Submits bid above stated cap"

INFERENCE — "When the counterpart publicly does P, it likely reveals hidden information H."
  Fields:
    public_action      : A specific observable action or announcement by the counterpart.
    hidden_information : A PRIVATE VARIABLE the counterpart holds that this action reveals
                         (their valuation range, budget constraint, hand state).

  Bullet rules:
    - hidden_information must be a PRIVATE VARIABLE — not a motivation or future plan.
      BAD: "Strategic motivation to suppress competition" (motivation)
      BAD: "Will bid higher than stated" (future action)
      GOOD: "Private valuation is constrained to ≤3"
      GOOD: "Budget ceiling prevents bidding above 4"

== QUALITY GATE — only generate a delta if ALL conditions are met ==

1. OUTCOME-ALTERING  : Would knowing this change my bid or move in a future game?
2. REPEATABLE        : Could this trigger appear again in a future game?
3. ANCHOR PURITY     : Is situation / public_action a pure trigger with no counterpart-response embedded?
4. PRIVATE-STATE PURITY (INFERENCE only): Is hidden_information a private variable — not a motive or plan?
5. NOVEL OR INCREMENTAL: Is this a genuinely new pattern, or a confirmed repeat of an existing one?

If you cannot answer YES to all five, discard the observation.

== ANCHOR REUSE RULE ==

Read the CURRENT COUNTERPART MEMORY JSON.
- If the trigger you observed matches an existing entry, use that entry's `id` as the `anchor_id` in your delta.
- Only use NEW_ENTRY if the trigger is genuinely different from all existing entries.

== DELTA OPERATIONS ==

ADD_COUNT     — An existing variant was observed again (increment its count).
                Required: type, op, anchor_id, action_id OR hidden_info_id, count

ADD_VARIANT   — A new variant was observed for an existing anchor.
                Required: type, op, anchor_id, action_description OR hidden_information_description, count

NEW_ENTRY     — New anchor + first variant (only if NO existing anchor matches this trigger).
                CONDITIONAL: type, op, situation, action_description, count
                INFERENCE  : type, op, public_action, hidden_information_description, count

MODIFY_ANCHOR — Refine the trigger text (game_state or public_action) of an existing entry.
                Required: type, op, anchor_id, new_anchor_text

MODIFY_VARIANT_DESC  — Improve the description text of an existing variant.
                       Required: type, op, anchor_id, action_id OR hidden_info_id, new_description

== OUTPUT FORMAT ==

Output ONLY a JSON array wrapped in ```json ... ``` fences.
Output an empty array [] if nothing from this game is worth memorizing.

```json
[
  {{
    "type": "CONDITIONAL",
    "op": "ADD_COUNT",
    "anchor_id": "c001",
    "action_id": "A",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "ADD_VARIANT",
    "anchor_id": "c001",
    "action_description": "Bids 2+ above publicly stated ceiling",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "NEW_ENTRY",
    "situation": "You opened with a bid below 5 and counterpart is in the submission phase",
    "action_description": "Counterpart submits a bid significantly higher than yours",
    "count": 1
  }},
  {{
    "type": "CONDITIONAL",
    "op": "MODIFY_ANCHOR",
    "anchor_id": "c002",
    "new_anchor_text": "Counterpart publicly announces a bid ceiling before submitting their bid"
  }}
]
```
"""
