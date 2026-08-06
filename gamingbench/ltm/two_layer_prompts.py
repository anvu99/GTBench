# gamingbench/ltm/two_layer_prompts.py
# This file contains all the prompt templates used by the Two-Layer Memory Agents.

# ---------------------------------------------------------
# IN-GAME PROMPTS
# ---------------------------------------------------------

# Used by EvidenceMemoryAgent to summarize the board state and intent before retrieving memories.
OBS_SUMMARIZE_PROMPT = """
You are analyzing a game state. 
Based on the current observation, please write a brief summary (1-3 sentences) of what the opponent seems to be doing. Focus on their strategic intent.

Note: Your generated summary will be used to search a database of past games to retrieve relevant strategic memories about this specific opponent. Make your summary descriptive enough to match similar historical situations.

Output exactly in strict JSON format like this:
```json
{
  "summary": "<your summary here>"
}
```
"""

# Used by ProactiveQueryAgent. Replaces OBS_SUMMARIZE_PROMPT. 
# It asks the agent to actively formulate a question about the opponent's strategy, injecting top-scoring past questions.
PQA_QUESTION_GEN_PROMPT = """
You are analyzing a game state.

Here are your top-performing strategic questions from past games against this opponent:
{top_questions}

Here is your current working memory (a short summary of what you already know and what you are currently exploring about this opponent):
{working_memory}

Based on the current observation, please do TWO things:
1. Write a brief summary (1-3 sentences) of what the opponent seems to be doing. Focus on their strategic intent.
2. Ask between 1 and {max_questions} distinct strategic questions about the opponent's historical behavior, patterns, or heuristics that you want answered to improve your play on this turn.

CRITICAL RULES FOR FORMULATING YOUR QUESTIONS:
- You are formulating queries to search a historical DATABASE about the opponent. You are NOT speaking to the opponent directly. Do not address the opponent as "you".
- Ask specifically about THIS OPPONENT and their past tendencies.
- DO NOT ask abstract or general game theory questions.
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
You are reviewing a completed game against an opponent.
Here is the full game trajectory:
{game_trajectory}

Your task is to extract objective, factual evidence (observations of the opponent's behavior) from this game.
Extract up to 5 key pieces of evidence. For each piece of evidence, provide:
- "content": A concrete, low-level factual description of EXACTLY what the opponent did that answers the question. You MUST explicitly note any mechanical constraints or game rules that attributed to the behavior (e.g., "The opponent bid 3 on round 2, but their private valuation was only 4, which constrained their maximum bid limit"). DO NOT write high-level strategic summaries here. Max 2 sentences.
- "observation": The strategic observation or hypothesis about the opponent's intent based on this evidence, which may potentially be true and helps relate this evidence to memory. Max 2 sentences.

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
You are managing a database of strategic memories about an opponent.
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
2. You do NOT need to force the memory into an overly generalized or rigid rule. A single memory can explicitly list multiple distinct cases, conditions, or causes that explain the opponent's behavior.
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

# Used by ProactiveQueryAgent during post-game log processing.
# Consolidates all in-game questions (both answered and unanswered) into a single batched review.
POST_GAME_QUESTION_REVIEW_PROMPT = """
During the game, you asked the following questions.
Some of these you retrieved via direct ID, and others were new questions matched via semantic search.

Here is the log of your questions:
{question_log}

Now that the game is over, review the full game trajectory:
{game_trajectory}

Your task is to review each question in the log based on how the opponent ACTUALLY played in the trajectory.
CRITICAL RULE: Carefully consider mechanical game constraints (e.g., maximum legal bids based on random private valuations, budget limits, forced actions) before attributing an action to a psychological strategy or betrayal. If an opponent was forced to make a seemingly hostile or passive move due to a game rule or poor random valuation, do NOT assume they broke an agreement maliciously.

You must assign a "tag" to each question. The allowed tags depend on the section:

For SECTION A (Direct Retrieval Questions):
- "tag" MUST be either "REINFORCE" or "MODIFY". (UNANSWERED is illegal because the memory already exists).
- If the retrieved memory accurately predicted the opponent's behavior, use "REINFORCE". Extract factual observations that support the memory.
- If the memory was FALSE/contradicted by actual play, OR if the agent explicitly provided "desired_additional_info" to request more detail, use "MODIFY". Extract factual evidence that either corrects the false assumption or provides the requested additional detail.

For SECTION B (New Questions):
- "tag" can be "REINFORCE", "MODIFY", or "UNANSWERED".
- If a "Driving Memory Content" is shown: you MUST use "REINFORCE" or "MODIFY". Prefer MODIFY over UNANSWERED. This means your extracted evidence will be used to update this existing driving memory so that it explicitly contains the correct information needed to answer the new Question.
- If no "Driving Memory Content" is shown: you MUST use "UNANSWERED" to create a new memory. Extract factual evidence that answers the question.

When extracting evidence, follow this rule:
- "content": A concrete, low-level factual description of EXACTLY what the opponent did that answers the question. You MUST explicitly note any mechanical constraints or game rules that attributed to the behavior (e.g., "The opponent bid 3 on round 2, but their private valuation was only 4, which constrained their maximum bid limit"). DO NOT write high-level strategic summaries here. Max 2 sentences.
- "observation": The strategic observation or hypothesis about the opponent's intent based on this evidence, which may potentially be true and helps relate this evidence to memory. Max 2 sentences.

Output exactly in strict JSON format. Your output must be a single JSON object containing a "question_reviews" array.
Each review must include the question_id so we can map it back.

Example:
```json
{{
  "question_reviews": [
    {{
      "question_id": "Q1",
      "tag": "MODIFY",
      "evidence": [
        {{
          "content": "<factual observation that corrects the memory or provides the requested additional info>",
          "observation": "<strategic observation>"
        }}
      ]
    }},
    {{
      "question_id": "Q2",
      "tag": "UNANSWERED",
      "evidence": [
        {{
          "content": "<factual observation that answers the previously unanswered question>",
          "observation": "<strategic observation>"
        }}
      ]
    }},
    {{
      "question_id": "Q3",
      "tag": "REINFORCE",
      "evidence": [
        {{
          "content": "<factual observation supporting the correct memory, or noting mechanical constraints>",
          "observation": "<neutral strategic observation>"
        }}
      ]
    }}
  ]
}}
```
"""

# ---------------------------------------------------------
# ProactiveQueryAgent: Post-Game Specialized Prompts
# ---------------------------------------------------------

PQA_MEMORY_MODIFY_PROMPT = """
You are tasked with surgically updating existing strategic memories about an opponent.

Here are the strategic memories that need modification, based on recent gameplay:
{modification_tasks}

Your task is to REWRITE the memory content for each task. A memory needs modification if it was either contradicted by new evidence, OR if the agent explicitly requested additional information ("desired_additional_info") that you must now bake into the memory.
You must write the new memory in a way that still preserves the old correct cases while incorporating the new insights, as all evidence represents factual observations.

CRITICAL RULES:
1. The new memory content must remain a concise strategic insight, strictly limited to AT MOST 4 sentences long.
2. It must directly answer the original question in light of the new evidence. If 'desired_additional_info' is provided, use the new as well as old evidences to answer that specific need, rather than writing instructions for future tracking.
3. NEVER output or modify the question of a memory. Each task shows you the question for context only. The question is IMMUTABLE. Your output JSON must contain ONLY "memory_id" and "memory_content" — never a "question" field.

Output exactly in strict JSON format:
```json
{{
  "modifications": [
    {{
      "memory_id": "<memory_id>",
      "memory_content": "<your rewritten memory here>"
    }}
  ]
}}
```
"""

PQA_UNANSWERED_SYNTHESIS_PROMPT = """
You are tasked with generating brand new strategic memories to answer specific questions about the opponent.
During recent games, the agent asked these unanswered questions.

Here are the questions along with the new factual evidence that helps answer them:
{synthesis_tasks}

Here is related historical evidence from past games (Layer 1 observations) that may provide additional context for all questions:
{related_evidence_list}

Your task is to synthesize this evidence into new strategic memories. If multiple questions ask about the same concept, you MUST group them together and generate a single comprehensive memory that answers them collectively. For conceptually distinct questions, generate separate memories.
If you group questions together, you MUST include all supporting evidence IDs from all the grouped questions in the "evidence_ids_used" array. Select ONLY the evidence IDs (from both the new and related lists) that actually support your conclusion.

CRITICAL RULES:
1. The memory_content must directly answer the question being asked. If "DESIRED ADDITIONAL INFO" is provided, it indicates the specific type of data the agent was looking for. You should use the provided evidence to answer that specific need. Do NOT write instructions or to-do lists for future tracking; only state the factual strategic conclusions you can draw from the current evidence.
2. It must be comprehensively written so that it supports and accounts for ALL of its selected evidence.
3. The memory_content must remain a concise strategic insight, strictly limited to AT MOST 4 sentences long.

Output exactly in strict JSON format.
Example:
```json
{{
  "new_memories": [
    {{
      "question": "<the question being answered>",
      "memory_content": "<new synthesized memory text>",
      "evidence_ids_used": ["ev_abc", "ev_xyz"]
    }}
  ]
}}
```
"""

# Injection prompt for EvidenceMemoryAgent
EVIDENCE_INJECTION_PROMPT = """\
=== HISTORICAL EVIDENCE ===
Based on your observation of the current game state, the following strategic insights about this opponent were retrieved from past games:

{retrieved_memories}

Use these insights to inform your next action.
===========================
"""

# Injection prompt specifically for ProactiveQueryAgent to inject into the step/chat prompt
PROACTIVE_INJECTION_BLOCK = """\
=== STRATEGIC MEMORY INJECTION ===
You asked the following questions this round. Use these insights to inform your next action.

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
- `desired_additional_info`: (string) if extra detail would be beneficial, OR if no memory was retrieved and you need specific quantitative/behavioral data to answer this in the future, describe exactly the detailed information you want. Leave empty ("") ONLY if the current information is fully sufficient.

Output exactly in strict JSON format. Your output must be a single JSON object containing a "strategy" object, an "assessments" array AND a "working_memory" string.

Your "strategy" object MUST be either a "follow" strategy or a "new" strategy.
If you are following an existing strategy from the STRATEGY MEMORY, output:
"strategy": { "type": "follow", "strategy_id": "<the strat_id>" }
If you are trying a completely new strategy, output:
"strategy": { "type": "new", "title": "<name>", "definition": "<details>", "success_criteria": "<conditions>", "neutral_criteria": "<conditions>", "failure_criteria": "<conditions>" }

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
      "desired_additional_info": "<If extra detail would be beneficial, describe exactly the detailed information you want. Leave empty ONLY if the current information is fully sufficient.>"
    },
    {
      "question_index": 2,
      "question_type": "new",
      "answered": true,
      "memory_conclusion": "<If answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.>",
      "driving_memory_id": "<If answered=true, provide the ID of the specific memory that helped you answer it. If answered=false, this MUST be null.>",
      "desired_additional_info": "<If extra detail would be beneficial, OR if no memory was retrieved and you need specific quantitative/behavioral data to answer this, describe exactly the detailed information you want. Leave empty ONLY if the current information is fully sufficient.>"
    }
  ],
  "working_memory": "<Write a NEW rolling summary (MAXIMUM 6 sentences) of the most important strategic insights you have learned and the questions you are actively exploring about this opponent.>"
}
```
"""

STRATEGY_INJECTION_BLOCK = """\
=== STRATEGY MEMORY ===
Top strategies you have used in past games (ranked by performance):

{top_strategies}
======================
"""

STRATEGY_SCORING_PROMPT = """
You are reviewing a completed game to evaluate the effectiveness of the strategies you employed.

Here are the strategies you attempted during this game (along with their scoring criteria):
{strategies_to_score}

Here is the full game trajectory:
{game_trajectory}

Your task is to evaluate each strategy based STRICTLY on its own defined success, neutral, and failure criteria.

Output exactly in strict JSON format:
```json
{{
  "scores": [
    {{
      "strategy_id": "<strategy_id_or_temp_key>",
      "score": "<must be exactly one of: success, neutral, failure>",
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
For each group of similar strategies, you must select exactly ONE to keep (preferring the one with the best score: success > neutral > failure).
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
