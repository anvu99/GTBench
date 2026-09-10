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

OPPONENT line: A brief statistic that justifies the DO/INSTEAD recommendation.
  Include only the evidence most relevant to that recommendation and total observation count.
  EXAMPLE: "Defects in 61% of cases (38 obs)" or "Complies 80% of the time (14 obs)"

== OUTPUT FORMAT ==

=== STRATEGY BRIEF (Based on {games_observed} games) ===

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
{tendency_block}

Use this opponent profile strategically:
- CONDITIONAL entries: when you recognise the stated trigger in the current game state, \
use the action frequency distribution to anticipate what the opponent will do next.
- INFERENCE entries: when you observe the stated public action from the opponent, \
update your belief about their private information using the listed hidden-state distribution.\
"""

# =============================================================================
# Empty-state block (shown when no games have been observed yet)
# =============================================================================

TENDENCY_EMPTY_BLOCK = """\
=== OPPONENT PROFILE ===
No games observed against this opponent yet. No tendency data available.
======================="""
