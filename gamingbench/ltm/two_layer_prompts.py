# gamingbench/ltm/two_layer_prompts.py
# This file contains all the prompt templates used by the Two-Layer Memory Agents.

# ---------------------------------------------------------
# IN-GAME PROMPTS
# ---------------------------------------------------------

# Used by EvidenceMemoryAgent to summarize the board state and intent before retrieving memories.
OBS_SUMMARIZE_PROMPT = """
You are analyzing a game state. 
Based on the current observation, please write a brief summary (1-3 sentences) of what the other player seems to be doing. Focus on their strategic intent.

Note: Your generated summary will be used to search a database of past games to retrieve relevant strategic memories about this specific other player. Make your summary descriptive enough to match similar historical situations.

Output exactly in strict JSON format like this:
```json
{
  "summary": "<your summary here>"
}
```
"""

# Used by ProactiveQueryAgent. Replaces OBS_SUMMARIZE_PROMPT. 
# It asks the agent to actively formulate a question about the other player's strategy, injecting top-scoring past questions.
PQA_QUESTION_GEN_PROMPT = """
You are analyzing a game state.

Here are your top-performing strategic questions from past games against this other player:
{top_questions}

Here is your current working memory (a short summary of what you already know and what you are currently exploring about this other player):
{working_memory}

Based on the current observation, please do TWO things:
1. Write a brief summary (1-3 sentences) of what the other player seems to be doing. Focus on their strategic intent.
2. Ask between 1 and {max_questions} distinct strategic questions about the other player's historical behavior, patterns, or heuristics that you want answered to improve your play on this turn.

CRITICAL RULES FOR FORMULATING YOUR QUESTIONS:
- TIP: These questions are the ONLY way you can retrieve information from your large player profile database. Think of what information you want to know about the other player that can help you improve your strategy and maximize utility. Generate questions that are clear and direct to get the exact information you want.
- You are formulating queries to search a historical DATABASE about the other player. You are NOT speaking to the other player directly. Do not address the other player as "you".
- Ask specifically about THIS OTHER PLAYER. Since you only have 1 opponent or peer, always refer to them generally as "the opponent" or "the other player" in your question (e.g., "How often does the opponent...").
- DO NOT ask abstract or general game theory questions.
- CRITICAL: The database only stores objective information that can be perceived post-game from the game trajectory. DO NOT ask for subjective advice, strategies, or instructions on how you should play (e.g., "What should I do?"). You CAN ask for measurable outcomes of specific actions (e.g., "What is the opponent's response if I do X?").
- For each question, if it is semantically the same as one of the top-performing questions above, you are strongly encouraged to copy it exactly for best retrieval. Set source_memory_id to its exact memory ID (e.g., "mem_abc123"). 
- If you are asking a completely novel question, set source_memory_id to null.

Output exactly in strict JSON format like this:
```json
{{
  "summary": "<your summary here>",
  "questions": [
    {{
      "question": "<your first question here>",
      "source_memory_id": "<mem_id or null>"
    }}
  ]
}}
```
"""



# ---------------------------------------------------------
# POST-GAME PROMPTS
# ---------------------------------------------------------

# Used by both agents at the end of a game (or end of a batch).
# Extracts factual, objective evidence (Layer 1 data) from the raw game trajectory.
EVIDENCE_EXTRACT_PROMPT = """
You are reviewing a completed game against another player.
Here is the full game trajectory:
{game_trajectory}

Your task is to extract objective, factual evidence (observations of the other player's behavior) from this game.
Extract up to 5 key pieces of evidence. For each piece of evidence, provide:
- "content": A concrete, low-level factual description of EXACTLY what the other player did that answers the question. You MUST explicitly note any mechanical constraints or game rules that attributed to the behavior (e.g., "The other player bid 3 on round 2, but their private valuation was only 4, which constrained their maximum bid limit"). DO NOT write high-level strategic summaries here. Max 2 sentences.
- "observation": The strategic observation or hypothesis about the other player's intent based on this evidence, which may potentially be true and helps relate this evidence to memory. Max 2 sentences.

Output strictly in JSON format like this:
```json
{{
  "evidence": [
    {{
      "content": "<factual observation>",
      "observation": "<strategic observation>"
    }}
  ]
}}
```
"""

# The core unified logic for linking Evidence -> Memory.
# Used by both agents whenever new evidence is generated (either from trajectory extraction or question answering).
# It receives the new evidence + top-K relevant existing memories, and decides to either:
# 1. Update an existing memory
# 2. Create a new memory
ROUTE_AND_MODIFY_PROMPT = """
You are managing a database of strategic memories about another player.
You have just extracted the following new pieces of evidence:
{new_evidence_list}

Here are the most relevant existing memories from your database (each with their associated evidence snippets):
{existing_memories}

Your task is to process the new pieces of evidence. For each piece of evidence (or group of related evidence), you must decide whether to append it to an existing memory (modifying the memory to incorporate the new insight) or to create a completely new memory.

You can group multiple related new evidence IDs together into a single decision if they point to the same strategic insight.

If you append to an existing memory, you must rewrite its content to comprehensively reflect BOTH the new evidence and the existing evidence because all evidence represents factual, observable behavior.
If you create a new memory, you must list the evidence IDs that support it (which MUST include the new evidence IDs, and optionally any other relevant evidence IDs shown above).

CRITICAL RULES FOR WRITING MEMORIES:
1. The `memory_content` must be comprehensively written so that it supports and accounts for ALL of its linked evidence (both the existing evidence and the newly added evidence).
2. You do NOT need to force the memory into an overly generalized or rigid rule. A single memory can explicitly list multiple distinct cases, conditions, or causes that explain the other player's behavior.
3. The `memory_content` must remain a concise strategic insight, strictly limited to AT MOST 4 sentences long.

Output exactly in strict JSON format. Your output must be a single JSON object containing a "routing_decisions" array. 

Example:
```json
{{
  "routing_decisions": [
    {{
      "routing": "ADD_TO_mem_123",
      "evidence_ids_used": ["ev_abc", "ev_def"],
      "memory_content": "<updated memory text>"
    }},
    {{
      "routing": "CREATE_NEW",
      "evidence_ids_used": ["ev_xyz", "ev_uvw"],
      "memory_content": "<new memory text>"
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# ProactiveQueryAgent: Stats Pool Phase Prompts
# ---------------------------------------------------------

"""
This file contains the multi-phase prompts used by the ProactiveQueryAgent 
for post-game memory and stat processing.

The phases are structured as follows:
- Phase A (Proposal): STAT_PROPOSAL_PROMPT generates draft memories and requests new stats.
- Phase B (Batched Definition): STAT_DEFINITION_PROMPT takes proposed stats across all games in a batch and decides whether to inherit existing ones or create new ones.
- Phase 1 (Update): STAT_UPDATE_PROMPT processes game trajectory to update numerical stat values.
- Phase C (Content Check): MEMORY_CONTENT_UPDATE_PROMPT modifies existing memory texts and manages stat eviction (10-stat cap).
- Phase D (Finalization): NEW_MEMORY_FINALIZATION_PROMPT finalizes new memories and manages stat eviction.

All prompts receive `game_rules` to provide environmental context.
"""

# ---------------------------------------------------------
# Phase A: Propose new stats and memories
# ---------------------------------------------------------
STAT_PROPOSAL_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

You are tasked with proposing statistical trackers for the agent's strategic memories based on queries generated during recent games.
Some queries were brand new questions (labeled with `[N_#]`), and some were requests for additional information for existing memories (labeled with `[mem_#]`).

Here is the log of queries:
{unanswered_questions}

Here is the full game trajectory:
{game_trajectory}

Your task is to review the trajectory and address each query.
- DO NOT generate memory content or answer the questions directly. 
- ONLY propose stats that would help the agent track the requested concept or answer the question over time, strictly based on the explicit question and the desired additional information requested. Do not propose stats to track things that are not explicitly requested in the desired information.
- First, review all new questions `[N_#]`. Identify any that ask about the exact same strategic concept or other player behavior. Select exactly ONE to keep from each duplicate group, and discard the rest.
- Second, generate a stat proposal for EVERY `[N_#]` you chose to keep, AND for EVERY existing memory request `[mem_#]`.

CRITICAL RULES:
- TIP: Your stat description must be a direct, clear, and concise human-readable explanation of what the stat tracks. It is used by the agent to understand the stat's semantic purpose.
- TIP: Your stat pseudocode will be persistent and used by the agent to update the stat in future games. Use logical conditions using the trajectory tags as your variables (e.g., `[Move] Opponent`, `Round X`, `<Y dices, Z value>`). The pseudocode should be easy to read and understand. It MUST be a single string (do NOT use nested JSON objects or lists for the pseudocode).
- TIP: Each stat type has a fundamental, simple data structure that strictly limits what it can store. Grouping a complex analysis into a single stat will fail. If you need to answer a complex question, break it down and use multiple simple stats that combining them will help you answer the question instead.
1. For `proposed_stats`, you may propose up to {max_stats_per_memory} trackers per query. The `type` MUST be one of:
   - COUNT: Tracks total occurrences of an event. (Data structure: simple `n` counter. Pseudocode MUST define a boolean `condition` to count).
   - RATE: Tracks a percentage. (Data structure: `count` and `total` counters. Pseudocode MUST define a boolean `trigger_condition` for when the denominator increments, and a boolean `success_condition` for when the numerator increments).
   - MEAN_VAR: Tracks average and variance. (Data structure: `n`, `sum`, and `sum_sq` counters. Pseudocode MUST define the `target_value` to extract).
   - DISTRIBUTION: Tracks a histogram. (Data structure: `buckets` dictionary mapping string categories to counts. Pseudocode MUST define the `category_key` to extract).
   - EXTREMUM: Tracks max/min bounds. (Data structure: `max` and `min` values. Pseudocode MUST define the `target_value` to extract).

Output exactly in strict JSON format:
```json
{{
  "new_questions_merge": {{
    "keep": ["<N_id_1>", "<N_id_2>"],
    "merge_groups": [
      {{
        "keep": "<N_id_to_keep>",
        "discard": ["<N_id_to_discard_1>"]
      }}
    ]
  }},
  "stat_proposals": [
    {{
      "source_id": "<The N_id or mem_id>",
      "question": "<The original question>",
      "desired_info": "<Any explicit desired additional info requested by the agent, or empty>",
      "memory_id": "<null if it is a new question (N_id), or the exact memory_id if it is an existing memory>",
      "proposed_stats": [
        {{
          "type": "<type>",
          "description": "<A concise human-readable explanation of what this stat tracks>",
          "pseudocode": "<pseudocode logic to calculate and update the data in each stat. It must be clear what update the agent needs to do to each data field>"
        }}
      ]
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# Phase A.5: Batched Memory Consolidation
# ---------------------------------------------------------
STAT_MEMORY_MERGE_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

Across multiple recent games, the agent proposed statistical trackers to answer several new strategic questions.
Because these proposals were generated independently per game, some of them might ask about the exact same strategic concept or other player behavior.

Here are the newly proposed questions and their stats:
{batched_proposals}

Your task is to DEDUPLICATE these proposals. 
- Identify proposals that ask about the exact same strategic concept or other player behavior.
- For each group of duplicate proposals, you must select exactly ONE index to keep, and discard the rest.
- When deciding which proposal to keep, choose the one whose question, description, and stat pseudocode are the most clear, direct, and helpful for maximizing the agent's utility in-game.
- Any proposal that is entirely unique and not similar to any other should also be kept.

Output exactly in strict JSON format:
```json
{{
  "keep": [0, 2],
  "merge_groups": [
    {{
      "keep": 1,
      "discard": [3, 4]
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# Phase B: Batched Stat Definition
# ---------------------------------------------------------
STAT_DEFINITION_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

You have proposed several new statistical trackers across multiple games.
For each proposed tracker, we searched the existing Stat Pool for semantically similar trackers.

{batched_proposals}

For each item above, you must decide the best action to take:
1. INHERIT: If the proposed tracker perfectly matches the intent of an existing stat from its Candidates list.
2. DEFINE_NEW: If the proposed tracker does not match any candidate and must be created as a new stat.
3. INHERIT_FROM_LOCAL: If multiple proposed trackers in this specific batch (the list above) are identical or highly semantically similar, you must only create ONE new stat for the first occurrence (using `define_new`), and then route all subsequent duplicates in the batch to inherit from that first definition (using `inherit_from_local` pointing to the target's `local_idx`).

Output exactly in strict JSON format:
```json
{{
  "decisions": [
    {{
      "local_idx": 0,
      "action": "inherit",
      "stat_id": "<stat_id_from_pool>"
    }},
    {{
      "local_idx": 1,
      "action": "define_new"
    }},
    {{
      "local_idx": 2,
      "action": "inherit_from_local",
      "target_local_idx": 1
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# Phase 1: Numerical Stat Update
# ---------------------------------------------------------
STAT_UPDATE_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

You are tasked with updating statistical trackers based on the latest game trajectory.

=== STAT UPDATE RULES ===
For RATE stats:
  Output delta "count" = times the success condition in the pseudocode occurred.
  Output delta "total" = times the trigger condition in the pseudocode was present.
  (If the context never appeared this game, output zeros for both.)

For COUNT stats:
  Output delta "n" = total times the condition in the pseudocode occurred.

For MEAN_VAR stats:
  Output delta "n" = total times the quantity was observed.
  Output delta "sum" = sum of the observed quantities.
  Output delta "sum_sq" = sum of the squared observed quantities.

For DISTRIBUTION stats:
  Output delta "buckets" = a dictionary mapping string bucket names to occurrence counts.

For EXTREMUM stats:
  Output delta "max" = maximum value observed (or null if none).
  Output delta "min" = minimum value observed (or null if none).

=== STAT EVALUATION PROTOCOL ===
You MUST use a <thinking> block before outputting your final JSON.
For each stat, go through the game trajectory, apply the pseudocode and finalize the delta for that stat. You may review the description to get more context on what the stat is tracking.

=== GAME TRAJECTORY ===
{game_trajectory}

=== STATS TO UPDATE ===
{stat_list}

Output exactly in strict JSON format:
```json
{{
  "stat_updates": [
    {{
      "stat_id": "<stat_id>",
      "deltas": {{ "<field>": <value> }}
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# Phase C: Existing Memory Content Update & Eviction
# ---------------------------------------------------------
MEMORY_CONTENT_UPDATE_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

You are tasked with maintaining the accuracy of strategic memories about another player.
After the most recent games, the statistical evidence supporting these memories was updated. Some memories may also have requested additional qualitative information.

For each memory below, decide if the memory content needs to be rewritten based on the new stats.

{changed_memories}

For each memory:
1. Review the updated stats, newly added stats, and any "desired additional info" requested by the agent.
2. Write a concise memory content that directly answers the question using these exact numbers. If the new numbers significantly change the conclusion, if the current content is vague, or if the agent requested additional info, you must rewrite it. If the current content is already perfectly accurate and no additional info was requested, do not update it.
3. The new memory content must remain a concise strategic insight, strictly limited to AT MOST 4 sentences long. Embed the exact numbers/percentages directly into the text. DO NOT generate actionable advice, instructions, or strategies for how the agent should play.
4. NEVER output or modify the question of a memory. The question is IMMUTABLE.
5. Your memory can track at most {max_stats_per_memory} stats. After all proposed stats are added, the memory will have the stat count listed. If this exceeds {max_stats_per_memory}, you MUST populate `evict_stat_ids` with the IDs of the least useful stats to remove, until the total is <= {max_stats_per_memory}. If you update the text to remove a reference to an evicted stat, do so here.

Output exactly in strict JSON format:
```json
{{
  "content_updates": [
    {{
      "memory_id": "<memory_id>",
      "update": true,
      "new_content": "<your rewritten memory here>",
      "evict_stat_ids": ["<stat_id>"]
    }},
    {{
      "memory_id": "<another_memory_id>",
      "update": false
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# Phase D: New Memory Finalization & Eviction
# ---------------------------------------------------------
NEW_MEMORY_FINALIZATION_PROMPT = """
You are acting as the strategic memory controller for an AI agent playing a game.
Game Rules:
{game_rules}

You have new questions about the other player that need to be answered and converted into strategic memories.
You proposed statistical trackers for these questions, which have now been tracked and evaluated over recent games.
Some questions also come with explicit "desired info" notes detailing exactly what the agent wanted to know.

Here are the questions and their resolved statistical data:
{new_questions}

For each question:
1. Review the resolved stats and their current numerical values.
2. Write a concise memory content that directly answers the question based on these exact numbers. 
3. The new memory content must remain a concise strategic insight, strictly limited to AT MOST 4 sentences long. Embed the exact numbers/percentages directly into the text.
4. Your memory can track at most {max_stats_per_memory} stats. If the number of resolved stats exceeds {max_stats_per_memory}, you MUST populate `evict_stat_ids` with the IDs of the least useful stats to remove.

Output exactly in strict JSON format:
```json
{{
  "finalized_memories": [
    {{
      "question": "<the question>",
      "update": true,
      "new_content": "<your concise 1-4 sentence answer incorporating the exact numbers>",
      "evict_stat_ids": ["<stat_id>"]
    }}
  ]
}}
```
"""

# Injection prompt for EvidenceMemoryAgent
EVIDENCE_INJECTION_PROMPT = """\
=== HISTORICAL EVIDENCE ===
Based on your observation of the current game state, the following strategic insights about this other player were retrieved from past games:

{retrieved_memories}

Use these insights to anticipate the other player's behavior and inform your next strategic action. Pay special attention to any quantified stats (e.g., frequencies, rates) provided to accurately assess risk and probabilities.
===========================
"""

# Injection prompt specifically for ProactiveQueryAgent to inject into the step/chat prompt
PROACTIVE_INJECTION_BLOCK = """\
=== STRATEGIC MEMORY INJECTION ===
You asked the following questions this round. Use these insights to anticipate the other player's behavior and inform your next strategic action. Pay special attention to any quantified stats (e.g., frequencies, rates) provided to accurately assess risk and probabilities.

{question_blocks}
=================================="""

IN_GAME_ASSESSMENT_SUFFIX = """

---
After your action or chat, append a JSON block assessing each of the questions labeled in the STRATEGIC MEMORY INJECTION above, AND declaring the strategy you are employing.

For [DIRECT RETRIEVAL] questions: output only `desired_additional_info`.
For [NEW QUESTION] questions: output `answered`, `memory_conclusion`, `driving_memory_id`, `desired_additional_info`.

Field definitions:
- `answered`: (boolean) true if the retrieved memories successfully answer the question, false otherwise.
- `memory_conclusion`: (string) if answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.
- `driving_memory_id`: (string or null) if answered=true, provide the exact ID (e.g. "mem_1234") of the specific memory that helped you answer it. If answered=false, this MUST be null.
- `desired_additional_info`: (string) This field is strictly for requesting what is still missing to fully answer the specific question you asked, or a more detailed answer that falls entirely within the exact scope of the original question. The underlying stats automatically update with more games, so DO NOT ask for a larger sample size. **CRITICAL:** Do NOT expand the scope of the original question or ask about different scenarios. If you want to explore different conditions, you must ask a completely NEW question next round instead of polluting this memory. Leave empty ("") if the current memory structure is sufficient.
- `definition`: (string) A detailed explanation of the strategy. It MUST clearly articulate the specific tactical action to be taken, the underlying intent/reason, AND the target outcome it is trying to achieve. Make the strategy description reusable; only include specific state information (like specific cards, private valuation, specific rounds,...) if that is a key component driving when to use the strategy.
- `success_criteria`: (string) Describe the EXPECTED SUCCESSFUL OUTCOME of this strategy. Write it as a description of what the game state will look like if everything goes right (e.g., "The other player responds as anticipated (e.g., falling for a bluff, or correctly interpreting a hint), allowing us to maximize utility"). The successful outcome MUST ultimately correlate with maximizing expected utility or achieving a positive payoff. Do not describe an outcome as a 'success' if it results in sub-optimal utility. MUST be objectively verifiable by an observer reading the CURRENT game trajectory after it finishes.
- `failure_criteria`: (string) Describe the ANTICIPATED FAILURE OUTCOME of this strategy. Write it as a description of how this strategy could backfire, fail due to the other player's unexpected actions, or go wrong (e.g., "The other player anticipates our maximum bid and outbids us, or misinterprets our hint and discards a critical card"). MUST be objectively verifiable by an observer reading the CURRENT game trajectory after it finishes.

Output exactly in strict JSON format. Your output must be a single JSON object containing a "strategy" object, an "assessments" array AND a "working_memory" string.

Your "strategy" object MUST be either a "follow" strategy or a "new" strategy.
If you are following an existing strategy from the STRATEGY MEMORY, output:
"strategy": { "type": "follow", "strategy_id": "<the strat_id>" }
If you are trying a completely new strategy, output:
"strategy": { "type": "new", "title": "<name>", "definition": "<details>", "success_criteria": "<conditions>", "failure_criteria": "<conditions>" }

Example:
```json
{
  "strategy": {
    "type": "follow",
    "strategy_id": "strat_abc123"
  },
  "assessments": [
    {
      "question_index": 1,
      "question_type": "direct",
      "desired_additional_info": "<Strictly state what is missing to answer the specific question, or a more detailed metric falling entirely within the question's exact scope. DO NOT ask for more samples. DO NOT expand the scope to different conditions. Leave empty ONLY if the current information is fully sufficient.>"
    },
    {
      "question_index": 2,
      "question_type": "new",
      "answered": true,
      "memory_conclusion": "<If answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.>",
      "driving_memory_id": "<If answered=true, provide the ID of the specific memory that helped you answer it. If answered=false, this MUST be null.>",
      "desired_additional_info": "<Strictly state what is missing to answer the specific question, or a more detailed metric falling entirely within the question's exact scope. DO NOT ask for more samples. DO NOT expand the scope to different conditions. Leave empty ONLY if the current information is fully sufficient.>"
    }
  ],
  "working_memory": "<Write a NEW rolling summary (MAXIMUM 6 sentences) of the most important strategic insights you have learned and the questions you are actively exploring about this other player.>"
}
```
"""

IN_GAME_ASSESSMENT_SUFFIX_NO_STRATEGY = """

---
After your action or chat, append a JSON block assessing each of the questions labeled in the STRATEGIC MEMORY INJECTION above.

For [DIRECT RETRIEVAL] questions: output only `desired_additional_info`.
For [NEW QUESTION] questions: output `answered`, `memory_conclusion`, `driving_memory_id`, `desired_additional_info`.

Field definitions:
- `answered`: (boolean) true if the retrieved memories successfully answer the question, false otherwise.
- `memory_conclusion`: (string) if answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.
- `driving_memory_id`: (string or null) if answered=true, provide the exact ID (e.g. "mem_1234") of the specific memory that helped you answer it. If answered=false, this MUST be null.
- `desired_additional_info`: (string) This field is strictly for requesting what is still missing to fully answer the specific question you asked, or a more detailed answer that falls entirely within the exact scope of the original question. The underlying stats automatically update with more games, so DO NOT ask for a larger sample size. **CRITICAL:** Do NOT expand the scope of the original question or ask about different scenarios. If you want to explore different conditions, you must ask a completely NEW question next round instead of polluting this memory. Leave empty ("") if the current memory structure is sufficient.

Output exactly in strict JSON format. Your output must be a single JSON object containing an "assessments" array AND a "working_memory" string. There is NO "strategy" field required.

Example:
```json
{
  "assessments": [
    {
      "question_index": 1,
      "question_type": "direct",
      "desired_additional_info": "<Strictly state what is missing to answer the specific question, or a more detailed metric falling entirely within the question's exact scope. DO NOT ask for more samples. DO NOT expand the scope to different conditions. Leave empty ONLY if the current information is fully sufficient.>"
    },
    {
      "question_index": 2,
      "question_type": "new",
      "answered": true,
      "memory_conclusion": "<If answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.>",
      "driving_memory_id": "<If answered=true, provide the ID of the specific memory that helped you answer it. If answered=false, this MUST be null.>",
      "desired_additional_info": "<Strictly state what is missing to answer the specific question, or a more detailed metric falling entirely within the question's exact scope. DO NOT ask for more samples. DO NOT expand the scope to different conditions. Leave empty ONLY if the current information is fully sufficient.>"
    }
  ],
  "working_memory": "<Write a NEW rolling summary (MAXIMUM 6 sentences) of the most important strategic insights you have learned and the questions you are actively exploring about this other player.>"
}
```
"""

STRATEGY_INJECTION_BLOCK = """\
=== STRATEGY MEMORY ===
Top strategies you have used in past games (ranked by performance). 
(Note: 'Success' count is how many times the Expected Successful Outcome occurred, and 'Failure' count is how many times the Anticipated Failure Outcome occurred. 'Avg Utility' is the empirical average payoff or winrate).

{top_strategies}
======================
"""

STRATEGY_SCORING_PROMPT = """
You are reviewing a completed game to evaluate the effectiveness of the strategies you employed.

Here are the strategies you attempted during this game (along with their scoring criteria):
{strategies_to_score}

Here is the full game trajectory:
{game_trajectory}

Your task is to evaluate each strategy based STRICTLY on its own defined success and failure criteria.

Output exactly in strict JSON format:
```json
{{
  "scores": [
    {{
      "strategy_id": "<strategy_id_or_temp_key>",
      "score": "<must be exactly one of: success, failure>",
      "rationale": "<brief 1-2 sentence explanation of why this score was assigned based on the game events>"
    }}
  ]
}}
```
"""

STRATEGY_MERGE_PROMPT = """
You are reviewing newly created strategies to merge highly similar ones and keep the memory bank clean.

Here are the new strategies proposed during the last game, along with their preliminary scores:
{new_strategies}

Your task is to identify strategies that are conceptually identical or highly similar variations of the same core idea.
For each group of similar strategies, you must select exactly ONE to keep (preferring the one with the highest Average Utility, or best success > failure score if tied).
Any strategy that is entirely unique and not similar to any other should also be kept.

Output exactly in strict JSON format:
```json
{{
  "keep": ["<id_1>", "<id_2>"],
  "merge_groups": [
    {{
      "keep": "<id_to_keep>",
      "discard": ["<id_to_discard_1>", "<id_to_discard_2>"]
    }}
  ]
}}
```
"""
