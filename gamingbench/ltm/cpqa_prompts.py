"""
gamingbench/ltm/cpqa_prompts.py

All prompt templates for ConsolidatedPQA v2 agent.
"""

# =============================================================================
# IN-GAME: Consolidated block injection wrapper
# =============================================================================

CPQA_MEMORY_INJECTION = """\
=== OPPONENT PROFILE ===
This is your memory about the opponent's reputation and past behavior. Use it
strategically to help you anticipate their actions and make better decisions.

{consolidated_block}
========================
"""

CPQA_MEMORY_INJECTION_EMPTY = """\
=== OPPONENT PROFILE ===
This is your memory about the opponent's reputation and past behavior. Use it
strategically to help you anticipate their actions and make better decisions.

No games observed against this opponent yet. No profile data available.
========================
"""

# =============================================================================
# IN-GAME: SUS suffix — appended to every action/chat prompt
# =============================================================================

CPQA_SUS_SUFFIX = """
---
After your action, append a <sus> block listing any opponent behaviors from this round that are NOT discussed in the OPPONENT PROFILE and that you find suspicious or anomalous.
DO NOT guess the opponent's psychological intent or state of mind. Describe ONLY the specific, observable behaviour.
Leave the block empty if nothing notable happened.

<sus>
- <behavior 1>
- <behavior 2>
</sus>
"""

# =============================================================================
# POST-GAME (Prompt 2): Memory confirmation — one call per game
# =============================================================================

CPQA_POST_GAME_CONFIRM_PROMPT = """
You are reviewing a completed game to update frequency counts in your
opponent memory.

=== GAME RULES ===
{game_rules}

=== FULL GAME TRAJECTORY ===
{game_trajectory}

=== OPPONENT MEMORY ENTRIES ===
{memory_list}

For each memory entry above:
1. Did the trigger/anchor condition occur in this game? (yes/no)
   - For UNCONDITIONAL memories: the trigger always applies, skip to step 2.
   - For CONDITIONAL memories: check if the described game state occurred.
   - For INFERENCE memories: check if the described public action was taken.
2. If the trigger occurred: which bullet(s) [A], [B], [C]... best describe
   what the opponent did? Select all that apply.
3. If the trigger occurred but NO existing bullet matches what you observed,
   describe the new behavioral variant in "new_behavior".

Rules:
- "new_behavior" should be a description of the unmatched variant,
  or null if all observed behavior matched an existing bullet.

Output strictly as JSON:
```json
{{
  "confirmations": [
    {{
      "mem_id": "mem_abc123",
      "trigger_occurred": true,
      "matched_bullets": ["A", "C"],
      "new_behavior": null
    }},
    {{
      "mem_id": "mem_def456",
      "trigger_occurred": true,
      "matched_bullets": [],
      "new_behavior": "Opponent explicitly rejected the proposal and bid at their maximum"
    }},
    {{
      "mem_id": "mem_ghi789",
      "trigger_occurred": false,
      "matched_bullets": [],
      "new_behavior": null
    }}
  ]
}}
```
"""

# =============================================================================
# POST-GAME (Prompt 3): SUS Extraction — one call per game (if sus generated)
# =============================================================================

CPQA_POST_GAME_SUS_EXTRACTION_PROMPT = """
You are analyzing a completed game to discover and structure new observations 
about the opponent's behavior.

=== GAME RULES ===
{game_rules}

=== FULL GAME TRAJECTORY ===
{game_trajectory}

=== EXISTING OPPONENT PROFILE ===
{memory_list}

=== IN-GAME NOTES (<sus> blocks) ===
{raw_sus_items}

Your task: analyze the FULL GAME TRAJECTORY and the IN-GAME NOTES to extract formal structured observations.
- Use the IN-GAME NOTES as hints for specific suspicious actions the agent noticed during play.
- ALSO, independently review the trajectory for any other key opponent behaviors or strategic decisions that are important to remember.
- CRITICAL: DO NOT generate observations for behaviors that are already clearly described in the EXISTING OPPONENT PROFILE. Only extract NEW behaviors, NEW variants, or notable deviations from their known profile.
- ACTIONABILITY FILTER: Filter all observations for strict actionability. A behavior is only worth remembering if it provides a repeatable weakness/sub-optimal tendency that can be exploited, OR a reliable tell that allows us to deduce their hidden information (e.g., budget, hand state). If a behavior cannot be strategically exploited in future games, DISCARD it.

For every notable behavior you discover (either from the notes or your own review), categorize it into one of three types and extract the required fields:

1. UNCONDITIONAL: General behavioral tendencies.
   - Requires field: "behavior" — a SPECIFIC, CONCRETE description of what the opponent actually did in this game.
   - Describe exactly what was observable: the specific action, chat content, or bidding choice. Do NOT generalize or abstract.

2. CONDITIONAL: Behavior triggered by a specific game state.
   - Requires fields: "game_state" (the trigger), "action" (what they did)
   - CRITICAL: "game_state" must describe only the STIMULUS — the situation or event that occurred. It must be completely neutral about what the opponent does next.
     BAD (reaction baked in): "Opponent announces a conservative cap and then violates it"
     GOOD (trigger only): "Opponent publicly announces a bid ceiling or cap before submitting"
   - "action" must be the purely OBSERVABLE REACTION — no interpretation or judgment. Describe what you could see, not what you conclude from it.
   - If a behavior is better described as "whenever X happens, the opponent does Y", it is CONDITIONAL.

3. INFERENCE: Private information or hidden state inferred from a public action.
   - Requires fields: "public_action" (what they did), "underlying_driver" (the hidden private state this reveals)
   - CRITICAL: "underlying_driver" must be a PRIVATE VARIABLE the opponent holds that is NOT directly observable — for example their private valuation range, a hard budget constraint, or a mechanical game limit.
     BAD (motivation/intent): "Strategic motivation to manipulate anchoring", "Tactical bluff to suppress competition"
     BAD (future action): "Will submit a higher bid than stated", "Plans to defect after cooperation"
     GOOD (private state): "Private valuation is constrained to ≤2", "Budget ceiling prevents bidding above 1"
   - If a behavior describes a future action or a psychological motivation rather than a private state, classify it as CONDITIONAL or discard it.

Output strictly as JSON:
```json
{{
  "structured_observations": [
    {{
      "type": "CONDITIONAL",
      "game_state": "Opponent proposed an equal split",
      "action": "They immediately reneged in the hidden action phase"
    }},
    {{
      "type": "UNCONDITIONAL",
      "behavior": "Aggressively bids on the first item regardless of value"
    }}
  ]
}}
```
"""


# =============================================================================
# BATCH (Step A): Dedup + edge creation — one prompt per observation type
# =============================================================================

CPQA_BATCH_DEDUP_EDGE_PROMPT_UNCONDITIONAL = """\
You are managing an opponent behavior graph. New behavioral observations have
been collected from {n_games} recent game(s) against this opponent.

=== GAME RULES ===
{game_rules}

=== NEW STRUCTURED OBSERVATIONS ===
{new_observations_block}

=== CANDIDATE EXISTING MEMORY COMPONENTS ===
Each component below was retrieved by semantic similarity to one or more new
observations.
{candidate_memories}

=== YOUR TASKS ===

All observations in this batch are UNCONDITIONAL — they describe general behavioral
tendencies that apply regardless of specific game state.

TASK 1 — DEDUPLICATION (within new observations):
Identify any new observations that describe the same underlying behavioral tendency,
even if worded differently or captured from different game instances.
For each such group, keep the most comprehensive and generalizable version and merge
the others into it (sum their counts).
The bar for merging is SEMANTIC EQUIVALENCE of the core tendency, not identical wording.

TASK 2 — EDGE CREATION:
Connect two observations (or an observation to an existing component) if they describe
the SAME behavioral TOPIC or DIMENSION — even if their specific manifestations are
different or even contradictory.

Examples of observations that SHOULD be connected (same topic, different variants):
  - "Usually bids conservatively to preserve margin" AND
    "Occasionally bids near-maximum when sensing an easy win"
    → both describe bidding aggressiveness; they become separate bullets in one component.

Examples of observations that should NOT be connected (genuinely different topics):
  - "Bids conservatively" AND "Uses cooperative chat framing to lower expectations"
    → different behavioral dimensions; they should be separate components.

The key question is: "Would a reader naturally think of these as different facets of the
same behavioral pattern, or as separate and unrelated patterns?" If the former, connect.


Edge target rules:
- Use "target": "mem_abc123" to connect to an EXISTING memory component.
- Use "target": "new_002" to connect to ANOTHER NEW observation in this batch.
  (Use this when two new observations clearly belong to the same component topic
   but no existing component covers it yet — especially common on early batches.)

Grouping rules:
- New observations connected to each other (via new→new edges) will be
  synthesized together into one new memory component.
- New observations connected to an existing component will be added to it.
- New observations with no edges start their own new singleton component.

Output strictly as JSON:
```json
{{
  "surviving_observations": [
    {{
      "id": "new_001",
      "merged_from": ["new_003", "new_007"],
      "count": 2,
      "type": "UNCONDITIONAL",
      "behavior": "..."
    }},
    {{
      "id": "new_002",
      "merged_from": [],
      "count": 1,
      "type": "UNCONDITIONAL",
      "behavior": "..."
    }}
  ],
  "edges": [
    {{
      "source": "new_001",
      "target": "mem_abc123",
      "reason": "same general behavioral tendency"
    }},
    {{
      "source": "new_002",
      "target": "new_005",
      "reason": "both describe the same underlying pattern"
    }}
  ]
}}
```
"""

CPQA_BATCH_DEDUP_EDGE_PROMPT_CONDITIONAL = """\
You are managing an opponent behavior graph. New behavioral observations have
been collected from {n_games} recent game(s) against this opponent.

=== GAME RULES ===
{game_rules}

=== NEW STRUCTURED OBSERVATIONS ===
{new_observations_block}

=== CANDIDATE EXISTING MEMORY COMPONENTS ===
Each component below was retrieved by semantic similarity to one or more new
observations.
{candidate_memories}

=== YOUR TASKS ===

All observations in this batch are CONDITIONAL — each describes a behavior triggered
by a specific game state. A CONDITIONAL observation has:
  - game_state: the trigger condition
  - action: what the opponent did when that trigger fired

TASK 1 — DEDUPLICATION (within new observations):
Identify observations that share the SAME game state trigger AND describe the same
resulting action. Merge them (sum their counts), keeping the most general description.
The bar for merging is: same trigger + same action pattern, not just topically related.
DO NOT merge observations that have different triggers, even if the resulting actions
look similar.

TASK 2 — EDGE CREATION:
Connect two observations (or an observation to an existing component) if and only if
they share the SAME game_state trigger (even if the resulting actions differ — those
become different bullets in the same component during synthesis).
DO NOT connect observations with different triggers. Different triggers must become
separate memory components.

Edge target rules:
- Use "target": "mem_abc123" to connect to an EXISTING memory component.
- Use "target": "new_002" to connect to ANOTHER NEW observation in this batch.
  (Use this when two new observations clearly belong to the same component topic
   but no existing component covers it yet — especially common on early batches.)

Grouping rules:
- New observations connected to each other (via new→new edges) will be
  synthesized together into one new memory component.
- New observations connected to an existing component will be added to it.
- New observations with no edges start their own new singleton component.

Output strictly as JSON:
```json
{{
  "surviving_observations": [
    {{
      "id": "new_001",
      "merged_from": ["new_003", "new_007"],
      "count": 2,
      "type": "CONDITIONAL",
      "game_state": "...",
      "action": "..."
    }},
    {{
      "id": "new_002",
      "merged_from": [],
      "count": 1,
      "type": "CONDITIONAL",
      "game_state": "...",
      "action": "..."
    }}
  ],
  "edges": [
    {{
      "source": "new_001",
      "target": "mem_abc123",
      "reason": "same game state trigger"
    }},
    {{
      "source": "new_002",
      "target": "new_005",
      "reason": "both triggered by the same game state"
    }}
  ]
}}
```
"""

CPQA_BATCH_DEDUP_EDGE_PROMPT_INFERENCE = """\
You are managing an opponent behavior graph. New behavioral observations have
been collected from {n_games} recent game(s) against this opponent.

=== GAME RULES ===
{game_rules}

=== NEW STRUCTURED OBSERVATIONS ===
{new_observations_block}

=== CANDIDATE EXISTING MEMORY COMPONENTS ===
Each component below was retrieved by semantic similarity to one or more new
observations.
{candidate_memories}

=== YOUR TASKS ===

All observations in this batch are INFERENCE — each infers a hidden state or private
constraint from a specific observable public action. An INFERENCE observation has:
  - public_action: the specific observable action taken by the opponent
  - underlying_driver: the hidden state or constraint this action reveals

TASK 1 — DEDUPLICATION (within new observations):
Identify observations that describe the SAME specific public action (even if worded
differently), AND infer the same underlying hidden driver. Merge them (sum their
counts), keeping the most comprehensive version.
The bar for merging is: same public action class + same deduced hidden state.

TASK 2 — EDGE CREATION:
Connect two observations (or an observation to an existing component) ONLY if they
share the SAME specific public action class. This means the opponent performed the
same type of observable action (e.g., "announces intended bid in pre-bid chat").
CRITICAL: Do NOT connect observations just because they are topically related or
belong to the same behavioral theme. The PUBLIC ACTION must be semantically the
same observable event. Different public actions that reveal different hidden states
must start separate memory components.

Edge target rules:
- Use "target": "mem_abc123" to connect to an EXISTING memory component.
- Use "target": "new_002" to connect to ANOTHER NEW observation in this batch.
  (Use this when two new observations clearly belong to the same component topic
   but no existing component covers it yet — especially common on early batches.)

Grouping rules:
- New observations connected to each other (via new→new edges) will be
  synthesized together into one new memory component.
- New observations connected to an existing component will be added to it.
- New observations with no edges start their own new singleton component.

Output strictly as JSON:
```json
{{
  "surviving_observations": [
    {{
      "id": "new_001",
      "merged_from": ["new_003"],
      "count": 2,
      "type": "INFERENCE",
      "public_action": "...",
      "underlying_driver": "..."
    }},
    {{
      "id": "new_002",
      "merged_from": [],
      "count": 1,
      "type": "INFERENCE",
      "public_action": "...",
      "underlying_driver": "..."
    }}
  ],
  "edges": [
    {{
      "source": "new_001",
      "target": "mem_abc123",
      "reason": "same public action class"
    }},
    {{
      "source": "new_002",
      "target": "new_005",
      "reason": "both inferred from the same type of public action"
    }}
  ]
}}
```
"""

# Backwards-compatible alias
CPQA_BATCH_DEDUP_EDGE_PROMPT = CPQA_BATCH_DEDUP_EDGE_PROMPT_UNCONDITIONAL

STEP_A_PROMPTS = {
    "UNCONDITIONAL": CPQA_BATCH_DEDUP_EDGE_PROMPT_UNCONDITIONAL,
    "CONDITIONAL":   CPQA_BATCH_DEDUP_EDGE_PROMPT_CONDITIONAL,
    "INFERENCE":     CPQA_BATCH_DEDUP_EDGE_PROMPT_INFERENCE,
}

# =============================================================================
# BATCH (Step B): Memory synthesis — one prompt per component type
# =============================================================================

CPQA_MEMORY_SYNTHESIS_PROMPT_UNCONDITIONAL = """\
You are synthesizing a structured memory entry about an opponent's behavior.

=== GAME RULES ===
{game_rules}

=== NEW OBSERVATIONS TO INTEGRATE ===
Observations assigned to this component:
{component_nodes}

=== EXISTING MEMORY FOR THIS COMPONENT (if any) ===
{existing_memory}

Your task: produce an updated UNCONDITIONAL memory entry.

An UNCONDITIONAL memory tracks a behavioral tendency that the opponent exhibits
regardless of specific game state.

IMPORTANT: The observations you receive are CONCRETE and SPECIFIC (e.g., "Claimed
valuation of 5 but bid 7", "Proposed a bid cap of 3 in chat"). Your job in Step B
is to ABSTRACT UP: identify what OBSERVABLE DIMENSION or BEHAVIORAL CATEGORY these
specific observations all belong to, and name it as such. The category exists to
organize the distribution of variants — specific instances become bullets.

The memory entry must have:
1. NAME — a concise title naming the OBSERVABLE BEHAVIORAL CATEGORY, never a judgment or conclusion.
   Ask yourself: "What dimension of behavior am I tracking?" — the answer is the name.   BAD names (conclusions): "Deceptive Signaling", "Hyper-Cooperative Yielding Tendency", "Aggressive Bluffing"
   GOOD names (categories): "Pre-Bid Chat Messaging Style", "Bid Magnitude Relative to Valuation", "Post-Round Chat Behavior"
   The name must be broad enough that both cooperative AND adversarial variants of this behavior fit inside it.
2. DESCRIPTION — one neutral sentence: "Tracks how the opponent [observable action category] across games."
   Do NOT write a conclusion into the description. It must be a neutral framing of what dimension is being tracked.
   BAD: "The opponent habitually employs misleading communications to pressure opponents."
   GOOD: "Tracks the style and content of the opponent's pre-bid chat messages across rounds."
3. TYPE — UNCONDITIONAL (fixed).
4. ANCHOR — leave as "" (not used for UNCONDITIONAL).
5. BULLETS — one bullet per DISTINCT behavioral variant of this tendency.
   Bullets must together represent the FULL DISTRIBUTION of observed variants — including both cooperative and adversarial manifestations if both have been observed. Do NOT only capture the most salient or dramatic variant.

   BULLET RULES:
   Each bullet = a meaningfully DIFFERENT way this tendency manifests.
   DO NOT create separate bullets for observations worded differently but describing
   the same underlying pattern — merge them into one bullet (sum their counts).
   A new bullet is justified ONLY when the variant represents a genuinely different
   magnitude, context, or outcome.

   When integrating new observations with existing bullets:
   - If a new observation matches the underlying behavior of an existing bullet,
     add its count to that bullet. Do not create a new bullet.
   - Only create a new bullet for genuinely new variants.
   - Preserve existing bullet IDs. New ones get the next letter.

   DEFENSE AGAINST MIS-GROUPING:
   Step A may occasionally group observations from different topics into this
   component by mistake. If you notice that some observations clearly do NOT belong
   to the same behavioral topic as the majority:
   - If EXISTING MEMORY is provided above: preserve that memory's established topic
     as the definitive anchor. Incoming observations that don't fit it are outliers
     — discard them regardless of their count.
   - If NO existing memory: anchor on the MAJORITY topic (highest total count).
   - In either case, do NOT force-fit outlier observations into bullets.
     It is better to leave a minor observation out than to pollute the memory
     with an unrelated behavior.

Output strictly as JSON:
```json
{{
  "name": "...",
  "description": "...",
  "type": "UNCONDITIONAL",
  "anchor": "",
  "bullets": [
    {{"id": "A", "description": "...", "count": 5}},
    {{"id": "B", "description": "...", "count": 2}}
  ]
}}
```
"""

CPQA_MEMORY_SYNTHESIS_PROMPT_CONDITIONAL = """\
You are synthesizing a structured memory entry about an opponent's behavior.

=== GAME RULES ===
{game_rules}

=== NEW OBSERVATIONS TO INTEGRATE ===
Observations assigned to this component:
{component_nodes}

=== EXISTING MEMORY FOR THIS COMPONENT (if any) ===
{existing_memory}

Your task: produce an updated CONDITIONAL memory entry.

A CONDITIONAL memory tracks behavior the opponent exhibits in response to a specific
game state trigger. It has ONE anchor (the trigger) and multiple bullets (the
different actions the opponent may take when that trigger fires).

The memory entry must have:
1. NAME — a concise title describing the TRIGGER SITUATION ONLY. The name must be
   derivable from the anchor alone, with no reference to any specific reaction.
   Test: cover the bullets and read only the name — it should NOT imply any particular outcome.
   BAD names (reaction baked in): "Violation of Stated Bid Ceiling", "Concession to High Valuation Signals"
   GOOD names (trigger-only): "Opponent Publicly Commits to a Bid Ceiling", "Opponent Receives High Valuation Signal"
2. DESCRIPTION — one neutral sentence: "Tracks the full distribution of opponent reactions when [trigger]."
   Do NOT bake a specific reaction or judgment into the description.
   BAD: "Tracks how the opponent systematically violates stated bid caps to secure wins."
   GOOD: "Tracks how the opponent's actual bid relates to their publicly stated ceiling."
3. TYPE — CONDITIONAL (fixed).
4. ANCHOR — the SINGLE game state trigger that unifies all observations. Must describe
   the STIMULUS only — completely neutral about what the opponent does next.
   BAD anchor: "Opponent announces conservative cap and deviates from it"
   GOOD anchor: "Opponent publicly announces a bid ceiling or cap before submitting"
   - If the new observations include multiple different triggers, use the trigger that
     appears most frequently and discard outliers. The anchor must be ONE specific
     trigger, not a compound or vague description.
   - If an existing anchor is already set, preserve it unless the majority of new
     observations clearly point to a more precise or general version.
5. BULLETS — one bullet per DISTINCT action the opponent takes when the trigger fires.

   BULLET RULES:
   Each bullet = a different ACTION the opponent took given the same trigger.
   Bullets must cover the FULL distribution — including cases where the opponent
   adheres, defects, partially complies, or does something unexpected.
   DO NOT describe the trigger again in the bullets — it is already in the anchor.
   DO NOT create separate bullets for the same action worded differently — merge them.
   A new bullet is justified ONLY when the opponent took a genuinely different action
   (e.g., "bid conservatively" vs "bid aggressively" are two distinct reactions).

   When integrating new observations with existing bullets:
   - Match each new observation's action to an existing bullet if semantically equivalent.
     Add its count to that bullet.
   - Create a new bullet only for a genuinely new action variant.
   - Preserve existing bullet IDs. New ones get the next letter.

   DEFENSE AGAINST MIS-GROUPING:
   Step A may occasionally group observations with different triggers into this
   component by mistake. If you notice that some observations have a clearly
   different trigger from the majority:
   - If EXISTING MEMORY is provided above: preserve that memory's established trigger
     as the definitive anchor. Incoming observations with a different trigger are
     outliers — do not include them as bullets, regardless of their count.
   - If NO existing memory: anchor on the MAJORITY trigger (highest total count).
   - In either case, do NOT force-fit observations with a different trigger.
     It is better to produce a clean, focused memory for the established trigger
     than to create a bloated, incoherent one.

Output strictly as JSON:
```json
{{
  "name": "...",
  "description": "...",
  "type": "CONDITIONAL",
  "anchor": "the single game state trigger",
  "bullets": [
    {{"id": "A", "description": "action variant A when trigger fires", "count": 5}},
    {{"id": "B", "description": "action variant B when trigger fires", "count": 2}}
  ]
}}
```
"""

CPQA_MEMORY_SYNTHESIS_PROMPT_INFERENCE = """\
You are synthesizing a structured memory entry about an opponent's behavior.

=== GAME RULES ===
{game_rules}

=== NEW OBSERVATIONS TO INTEGRATE ===
Observations assigned to this component:
{component_nodes}

=== EXISTING MEMORY FOR THIS COMPONENT (if any) ===
{existing_memory}

Your task: produce an updated INFERENCE memory entry.

An INFERENCE memory tracks what hidden private information or constraints can be
deduced from a specific observable public action by the opponent. It has ONE anchor
(the public action observed) and multiple bullets (the different hidden states that
could explain that action).

The memory entry must have:
1. NAME — a concise title describing the observable public action (not a conclusion about it).
2. DESCRIPTION — one sentence: "Tracks what [public action] reveals about the opponent's
   hidden private state (e.g., valuation range, budget constraint)."
3. TYPE — INFERENCE (fixed).
4. ANCHOR — the SINGLE specific public action that is being interpreted.
   - Generalize to the class of action if the observations describe the same type
     of action with slight surface variation (e.g., all involve "announcing bid in
     pre-bid chat" → generalize to that).
   - The anchor describes WHAT the opponent DOES publicly, not why.
5. BULLETS — one bullet per DISTINCT hidden private state that this public action may reveal.

   WHAT IS A VALID HIDDEN STATE:
   A hidden state is something the opponent privately knows that you CANNOT directly
   observe during the game — typically their valuation range, a hard budget/mechanical
   constraint, or a game-enforced ceiling on their bid.
   VALID: "Private valuation is constrained to ≤2", "Budget ceiling prevents bidding above 1"
   INVALID (motivation/intent): "Strategic motivation to manipulate anchoring", "Tactical bluff"
   INVALID (future action): "Will submit a higher bid than stated", "Plans to defect"

   BULLET RULES:
   Each bullet = a different HIDDEN PRIVATE STATE that explains the public action.
   DO NOT describe the public action in the bullets — it is already in the anchor.
   DO NOT create separate bullets for the same hidden state worded differently — merge them.
   DO NOT write motivations, psychological intent, or future actions as bullets.
   A new bullet is justified ONLY when the underlying private state is genuinely
   different (e.g., "valuation ≤1" vs "valuation 3-5" are two distinct private states).

   When integrating new observations with existing bullets:
   - Match each new observation's underlying_driver to an existing bullet if they
     describe the same hidden state. Add its count to that bullet.
   - Create a new bullet only for a genuinely new hidden driver/state.
   - Preserve existing bullet IDs. New ones get the next letter.

   DEFENSE AGAINST MIS-GROUPING:
   Step A may occasionally group observations with different public actions into this
   component by mistake. If you notice that some observations were triggered by a
   clearly different public action from the majority:
   - If EXISTING MEMORY is provided above: preserve that memory's established public
     action anchor. Incoming observations from a different public action are outliers
     — do not create bullets for them, regardless of their count.
   - If NO existing memory: anchor on the MAJORITY public action (highest total count).
   - In either case, do NOT force-fit observations with a different public action.
     It is better to produce a clean, focused memory for the established action
     than to mix unrelated public actions into one incoherent entry.

Output strictly as JSON:
```json
{{
  "name": "...",
  "description": "...",
  "type": "INFERENCE",
  "anchor": "the single public action being interpreted",
  "bullets": [
    {{"id": "A", "description": "hidden state / driver A that explains this action", "count": 5}},
    {{"id": "B", "description": "hidden state / driver B that explains this action", "count": 2}}
  ]
}}
```
"""

# Backwards-compatible alias
CPQA_MEMORY_SYNTHESIS_PROMPT = CPQA_MEMORY_SYNTHESIS_PROMPT_UNCONDITIONAL

STEP_B_PROMPTS = {
    "UNCONDITIONAL": CPQA_MEMORY_SYNTHESIS_PROMPT_UNCONDITIONAL,
    "CONDITIONAL":   CPQA_MEMORY_SYNTHESIS_PROMPT_CONDITIONAL,
    "INFERENCE":     CPQA_MEMORY_SYNTHESIS_PROMPT_INFERENCE,
}

# =============================================================================
# NOTE: No CPQA_CONSOLIDATION_PROMPT needed.
# Step C is handled programmatically: memories are sorted by total evidence
# and concatenated directly. See _run_step_C in consolidated_pqa_agent.py.
# =============================================================================
