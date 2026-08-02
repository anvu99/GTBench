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
# It asks the agent to actively formulate a question about the opponent's strategy.
QUESTION_GEN_PROMPT = """
You are analyzing a game state.
Based on the current observation, please do TWO things:
1. Write a brief summary (1-3 sentences) of what the opponent seems to be doing. Focus on their strategic intent.
2. Ask ONE strategic question about the opponent's historical behavior, patterns, or heuristics that you want answered to improve your play on this turn.

CRITICAL RULES FOR FORMULATING YOUR QUESTION:
- Ask specifically about THIS OPPONENT and their past tendencies.
- DO NOT ask abstract or general game theory questions (e.g., avoid "What is the Nash equilibrium?" or "How do rational players act?"). Your memory database only contains behavioral profiles of this specific opponent, not theoretical textbooks.

Note: Your generated question will be used as a search query against a database of past games to retrieve relevant strategic memories about this opponent. Formulate your question clearly so that it retrieves the best historical insights to help you answer it.

Output exactly in strict JSON format like this:
```json
{
  "summary": "<your summary here>",
  "question": "<your question here>"
}
```
"""

# Used by ProactiveQueryAgent *after* action generation.
# It forces the agent to assess whether the memories retrieved using its question actually answered it.
# This assessment is saved to the question_log for post-game processing.
ANSWER_ASSESSMENT_PROMPT = """
Earlier, you asked the following question about the opponent:
"{question}"

You were then provided with these memories retrieved from your database:
{retrieved_memories}

Did the retrieved memories help answer your question? 
Output exactly in strict JSON format like this (set "answered" to true or false):
```json
{{
  "answered": true, 
  "memory_conclusion": "<If answered=true, write the actual answer you derived from the memories. If answered=false, write a brief 1-2 sentence reason why the memories fell short.>"
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

If you append to an existing memory, you must rewrite its content to reflect the new evidence while preserving its previous insights.
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
Some of these you determined were answered by your retrieved memories, and some were not.

Here is the log of your questions:
{question_log}

Now that the game is over, review the full game trajectory:
{game_trajectory}

Your task is to review each question in the log based on how the opponent ACTUALLY played in the trajectory:
CRITICAL RULE: Carefully consider mechanical game constraints (e.g., maximum legal bids based on random private valuations, budget limits, forced actions) before attributing an action to a psychological strategy or betrayal. If an opponent was forced to make a seemingly hostile or passive move due to a game rule or poor random valuation, do NOT assume they broke an agreement maliciously.

1. For questions that were "ANSWERED IN-GAME: True": Verify if the Retrieved Memories accurately predicted the opponent's behavior in the trajectory, keeping your MEMORY CONCLUSION in mind. Set "correct" to true, false, or "undeterminable". If any memory was FALSE or contradicted by the opponent's actual moves, extract corrective evidence. If it is "undeterminable" (due to lack of evidence or game constraints masking intent), you can extract neutral evidence or leave the array empty.
2. For questions that were "ANSWERED IN-GAME: False": Set "correct" to null. Extract factual evidence from the trajectory that finally helps answer the question, informed by why it wasn't answered in-game (MEMORY CONCLUSION). Follow this rule:

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
      "correct": false,
      "evidence": [
        {{
          "content": "<factual observation that corrects the false assumption>",
          "observation": "<strategic observation>"
        }}
      ]
    }},
    {{
      "question_id": "Q2",
      "correct": null,
      "evidence": [
        {{
          "content": "<factual observation that answers the previously unanswered question>",
          "observation": "<strategic observation>"
        }}
      ]
    }},
    {{
      "question_id": "Q3",
      "correct": "undeterminable",
      "evidence": [
        {{
          "content": "<factual observation noting what the opponent did and the mechanical game constraints that masked their intent>",
          "observation": "<neutral strategic observation acknowledging the lack of definitive intent>"
        }}
      ]
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
PROACTIVE_INJECTION_PROMPT = """\
=== STRATEGIC MEMORY INJECTION ===
Earlier in this round, you analyzed the game state and asked the following strategic question:
"{question}"

Based on your question, the following memories from past games against this opponent were retrieved:
{retrieved_memories}

Use these insights to inform your next action.
==================================
"""
