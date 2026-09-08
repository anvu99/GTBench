"""
Prompt templates for SimpleTendencyAgent.

Two prompts:
  TENDENCY_REFLECTION_PROMPT  — per-game call, produces a delta list of behaviours observed
  TENDENCY_MERGE_PROMPT       — post-batch call, merges all deltas into the canonical block
"""

# ---------------------------------------------------------------------------
# Per-game reflection prompt
# Given the current tendency block + game trajectory, output a bullet list
# of behaviours observed in this single game with occurrence counts.
# ---------------------------------------------------------------------------
TENDENCY_REFLECTION_PROMPT = """\
You are a strategic observer reviewing a completed game against an opponent.

GAME RULES:
{game_rules}

CURRENT OPPONENT TENDENCY PROFILE (built from prior games):
{current_block}

GAME TRAJECTORY:
{game_history}

Your task: List every opponent behaviour, pattern, or tendency you directly observed in this game that would be strategically useful to know about in future games.

Rules:
- If the behaviour already appears in the tendency profile above, use the EXACT same label (copy it word-for-word). This allows accurate count tracking.
- If it is genuinely new and not covered by any existing entry, write a short action-guiding label that describes WHAT the opponent did AND in what situation. 
- For each behaviour, report how many times it occurred in THIS game only.
- Only include behaviours that are strategically meaningful — i.e. knowing this would change how you play against this opponent.
- If you observed no noteworthy behaviours, output an empty list.

Format — output ONLY a plain bullet list, one behaviour per line:
- <behaviour label>, occurrences: <N>
- <behaviour label>, occurrences: <N>
...
"""

# ---------------------------------------------------------------------------
# Batch merge prompt
# Given the pre-batch canonical block + all per-game deltas, produce an
# updated canonical block with counts merged in.
# ---------------------------------------------------------------------------
TENDENCY_MERGE_PROMPT = """\
You are updating an opponent tendency profile after a batch of completed games.

GAME RULES:
{game_rules}

CURRENT TENDENCY PROFILE (state before this batch):
{current_block}

PER-GAME OBSERVED DELTAS (one section per game):
{all_deltas}

Your task: Produce a single updated tendency profile that incorporates all the observed deltas.

Rules:
1. For each delta entry that matches an existing profile behaviour (same label or clearly the same pattern), add the delta occurrence count to the existing [seen: N] count.
2. For genuinely new delta behaviours not covered by any existing entry, add them as new entries starting at their observed occurrence count.
3. If delta entries from different games use slightly different wording for the same underlying behaviour, consolidate them under the clearest, most action-guiding label and combine their counts.
4. Increment the game count in the header by the number of games in this batch ({num_games} games).
5. If the total number of behaviour entries would exceed {max_behaviours}, you must drop entries to stay at or below {max_behaviours}. Drop the observations you judge to be least strategically useful.

Output ONLY the updated tendency profile block in this exact format (no extra text, no markdown fences):

=== OPPONENT TENDENCY PROFILE ===
Based on <N> games observed against this opponent.

- <behaviour label> [seen: <count>]
- <behaviour label> [seen: <count>]
...
=================================
"""

# ---------------------------------------------------------------------------
# In-game injection wrapper
# Wraps the stored tendency block for insertion into the observation prompt.
# ---------------------------------------------------------------------------
TENDENCY_INJECTION_BLOCK = """\
{tendency_block}
Use the above opponent tendency profile to inform your strategy. \
The [seen] count describes the number of occurrences of the behaviour.\
"""

# ---------------------------------------------------------------------------
# Empty-state block (shown when no games have been observed yet)
# ---------------------------------------------------------------------------
TENDENCY_EMPTY_BLOCK = """\
=== OPPONENT TENDENCY PROFILE ===
No games observed against this opponent yet. No tendency data available.
================================="""
