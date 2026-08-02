"""
Long-Term Memory (LTM) RAG Prompts

This module contains the core prompt templates used across the entire LTM RAG architecture.
It includes:
- Retrieval Injection Prompts: Used by `_build_prompts` to inject database knowledge into the live game.
- Summarization Prompts: Used by `_run_window_summarization` to compress recent game history.
- Gradient Engine Prompts: Used by the Reflection engines to critique completed games and propose structural updates.
- TGD Synthesis Prompts: Used by the Consolidation engines to merge the structural updates into the final text database.
"""

UNIFIED_MEMORY_PREAMBLE = """\
=== YOUR LONG-TERM MEMORY ===
You have the following memory database(s) built from experience in past games:{memory_overview}

How to use these memories:
- Consult Proactive and Reactive memories to evaluate potential moves or chat messages given the current game state. Review which memories apply and what they suggest.
- Then apply Verification memory: review the verification memory and apply the verification step for the moves or chat messages you plan to execute. If your candidate matches a check's Description, execute the Verification step before committing. Reject the candidate if verification fails.
"""

LTM_INJECTION_PROMPT = """\
=== REACTIVE MEMORY ===
Defend and counter-strategies against behavioral patterns, strategies, and reputation observed in the opponent across past games. When the opponent's behavior matches a signal's 'When' condition, execute the prescribed Policy to defend or counter.

Field guide:
- When: The game state or context that activates this signal. Match strictly against this — do NOT wait for the 'What' to happen first, it may be too late.
- What: What the opponent is likely attempting once this signal fires. Use for anticipating intent, not for deciding whether the signal fires.
- Policy: The action to execute when this signal fires.

{ltm_text}
=== END REACTIVE MEMORY ===
"""

SELF_LTM_INJECTION_PROMPT = """\
=== VERIFICATION MEMORY ===
Safety checks for your own recurring mistakes. After forming a candidate move, scan these checks and run Verification if the Description matches your situation.

Field guide:
- Description: The mistake or danger pattern to watch for in your own play.
- Verification: The concrete check to run before committing. If it shows the risk is real, reject the candidate and find an alternative.

{self_ltm_text}
=== END VERIFICATION MEMORY ===
"""

PROACTIVE_LTM_INJECTION_PROMPT = """\
=== PROACTIVE MEMORY ===
Offensive strategies and exploits you have learned across past games. Review these to identify attacking opportunities in the current game state.

Field guide:
- Objective: What this strategy aims to achieve.
- Policy: The concrete steps to execute the strategy.

{proactive_ltm_text}
=== END PROACTIVE MEMORY ===
"""

WINDOW_SUMMARIZE_PROMPT = """\
================================================================================
🛑 CRITICAL INSTRUCTION: REFLECTION PHASE 🛑
DO NOT MAKE A GAME MOVE! You are currently in the post-window REFLECTION phase, 
NOT the active game phase. Do not output any move actions (e.g. <C6>). 
Your ONLY task is to generate the summary below.
================================================================================

The last {K} moves just completed. Based on everything you have observed and decided this window, output EXACTLY the following two sections:

Game/Opponent summary: [A few sentences — key observations about the opponent's moves and behavior that you need for post-game evaluation. If chat is active, compare what the opponent communicated against their actual moves to surface any relevant patterns.]

Reasoning memory: [A few sentences — the core reasoning behind your own key moves this window.
  (1) Opponent signals: You MUST enumerate every Signal from the REACTIVE MEMORY DATABASE that you used and state (a) which specific move you played in response to its Policy and what the immediate resulting state was, (b) what do you think about this signal — did it effectively advance your objective (e.g., gave you an advantage or improved coordination), was it neutral, or did it harm your outcome. Also include any opponent signal whose trigger you observed but whose Policy you chose not to follow, explaining why.
  (2) Self signals: You MUST enumerate every Signal from the VERIFICATION MEMORY DATABASE that fired this window and state (a) whether you successfully followed the corrective Policy (for FLAW signals) or reinforced the effective tactic (for STRENGTH signals), and (b) what the resulting state was. Also include any self signal whose trigger you observed but whose Policy you did not follow, explaining why.
  (3) Proactive signals: You MUST enumerate every Signal from the PROACTIVE STRATEGY DATABASE that fired this window and state (a) whether you successfully followed the Policy, and (b) what the resulting state was. Also include any proactive signal whose trigger you observed but whose Policy you did not follow, explaining why.]

--- MEMORY DATABASE DEFINITIONS ---
- REACTIVE MEMORY DATABASE (Opponent Reputation): Contains defend and counter-strategies against this specific opponent's/partner's recurring behavioral patterns, strategies, and reputation across past matches. Used to defend against their strategies, exploit their weaknesses, and counter their traps.
- VERIFICATION MEMORY DATABASE: Contains your own recurring behavioral flaws, predictable tendencies, or proven successful tactics across past matches. Used to correct your own mistakes and prevent self-sabotage, or lean into your proven strengths.
- PROACTIVE STRATEGY DATABASE: Contains your overarching attacking, deceiving, and trapping strategies that span multiple turns. Used to execute offensive plans and exploit opponent predictability.

⚠ CRITICAL NOTE ON VISIBILITY: The FULL GAME HISTORY provided above is the POST-GAME GROUND TRUTH. Depending on the game's rules, it may reveal omniscient information (such as the opponent's private valuations, hidden cards, or secret budgets) that was STRICTLY HIDDEN from you during the live game. You MUST carefully review the game rules to determine which information is private vs public. When summarizing your reasoning or noting observations, you MUST NOT pretend that you could see any private information before it was revealed.

⚠ NOTE: If no REACTIVE MEMORY DATABASE or VERIFICATION MEMORY DATABASE is present, focus entirely on describing your own reasoning and observations this window.
"""

GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game against a specific opponent.
Your goal is to compare the agent's in-game observations against the GROUND TRUTH game history and produce a structured gradient report.

--- ✅ MATCH GROUND TRUTH (Full History) ---
These are the confirmed, objective move-by-move outcomes. Use these as the authoritative ground truth.
Note: The evaluated agent's own moves and identity are labeled as 'You' in both the GROUND TRUTH history and their window summaries.

⚠ CRITICAL NOTE ON VISIBILITY: This ground truth history may contain omniscient information (such as the opponent's private valuations, hidden cards, or secret budgets) that was STRICTLY HIDDEN from the agent during the live game, depending on the game rules. You MUST carefully review the game rules to determine which information is private vs public. When writing new memory signals or updating existing ones, your signal triggers ('When' conditions) MUST ONLY rely on information that the rules explicitly state is publicly visible to the agent during the game.

Reading the history:
{game_history_legend}
- [Chat]: Chat message sent that turn. (CRITICAL: ALWAYS verify if [Chat] lines actually exist in the game history below. If no chat lines exist, chat is NOT allowed in this game, and you MUST NOT generate any Chat-type signals).
- [Move]: The physical move executed after the position above.

{game_history}

--- AGENT'S IN-GAME OBSERVATIONS (Window Summaries) ---
These are the agent's own observations and strategic thoughts recorded during the game.
{window_summaries}

--- CURRENT LONG-TERM MEMORY (Prior) ---
This is what the agent believed about this opponent BEFORE the game started.
{current_ltm}
("(No memory yet)" means this is the first game)


You are building a GRADIENT REPORT for the agent's Reactive Memory Database.
The goal of this report is to improve the agent's knowledge of this partner/opponent so that in future games the agent can maximize its objective (e.g., win rate, joint score, or cooperative utility). Each proposed update should bring the database closer to an accurate, complete, and strategically actionable understanding of the opponent — closing knowledge gaps that, if resolved, would unlock better strategies. A high-quality report captures BOTH types of signals: (1) Negative signals — opponent/partner behaviors that led to failure, miscoordination, deception, or disadvantage; and (2) Positive/Exploitable signals — patterns in their play that led to successful coordination, or weaknesses that the agent successfully exploited.

--- WHAT IS THE REACTIVE MEMORY DATABASE? ---
This database stores defend and counter-strategies against THIS SPECIFIC OPPONENT's recurring behavioral patterns, strategies, reputation, and exploitable weaknesses, built up across multiple past games.
At inference time, the agent scans this database BEFORE each move/chat turn. When a signal's 'When' condition matches the current game state, the signal fires and its 'Policy' is injected directly into the agent's reasoning to guide a defensive or countering response.
This database is NOT for recording the agent's own tendencies or general game strategies — those belong in the Verification Memory Database and Proactive Strategy Database respectively.

⚠ REACTIVE PRIORITY DIRECTIVE: The Reactive Memory Database is EXCLUSIVELY for DEFENDING and COUNTERING. The agent already has dedicated memory for proactive offensive behavior (proactive memory) and self-consistency (verification memory). Therefore, you MUST prioritize discovering patterns where:
  1. The opponent triggers a specific, identifiable pattern or threat that requires a defensive or countering response.
  2. The opponent employs a deceptive or manipulative tactic that the agent must learn to recognize and neutralize.


--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* Type: Must be exactly "Chat" or "Action". This dictates when the signal fires. If Type is "Chat", the Policy MUST provide instructions supporting what the agent should say (e.g. how to deceive, coordinate, or extract info). If Type is "Action", the Policy MUST provide instructions on what physical move to execute.
* When: The anticipation trigger. This memory will be injected to warn the agent right BEFORE the opponent takes their turn. Therefore, this field MUST describe the static evidence (e.g., game state, chat log) strictly prior to the opponent's action. It cannot describe the opponent's action itself, otherwise it will fire too late. Describe everything that could plausibly have driven the opponent's decision based ONLY on the static evidence available at that moment. Do not infer triggers that were not directly observed. Maximum 4 sentences.
* What: The factual observation of what the opponent did. Write only what was directly observed. Never use conditional language (e.g., "as long as", "whenever", "unless") — those imply rules that may not have been tested. If a condition was not tested, state that explicitly. Maximum 4 sentences.
* Policy: The concrete action to execute. It is crucial that this policy is well-designed to be strictly actionable immediately when the 'When' condition fires. If there are different game states or edge cases where following this general policy would actively harm the agent (e.g., following a defensive rule during a winning race), you MUST explicitly list those exceptions and provide the alternative conditional policy for those specific cases. Maximum 6 sentences.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.
* Retrieved in rounds: Shows the exact rounds in this game where RAG retrieved and injected this signal into the agent's live reasoning context. If this field says "None (This memory was NOT retrieved in any round of this game)", it means RAG failed to retrieve this signal during live play. Pay special attention to unretrieved signals: if the agent would have significantly benefited from using a signal, you MUST prioritize updating its round anchors so RAG will retrieve it in similar future situations.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'When' condition. For example, your 'When' to generate an Action signal can be some chat behavior from the opponent. Or your 'When' for a Chat signal can be when the game state is in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

Analyze the game and propose updates using the following 5 tags:
- [REMOVE]: Identify a signal whose core observed behavior (When/What) is directly contradicted by the ground truth, or whose entire signal — even after potential modification — is net harmful to retain. Do NOT use [REMOVE] if only the Policy is wrong; use [MODIFY] to fix the Policy instead. Do NOT use [REMOVE] simply because a signal's trigger was not encountered this game — absence of evidence is not contradiction.
- [ADD]: Define a completely new observed behavior not yet covered by any signal in the current database. If an existing signal partially covers the behavior but one or more fields are incorrect, use [MODIFY] instead of adding a duplicate.
- [MODIFY]: Identify an existing signal worth keeping, but whose one or more fields are inaccurate, misleading, or too specific based on the current game evidence. Prefer [MODIFY] over [ADD] when the behavior is already partially captured by an existing signal. Only list the fields that are changing; omit all unchanged fields.
- [MERGE]: Identify two or more existing signals that are variations of the same underlying behavior. Use [MERGE] conservatively: do NOT merge signals if unifying the policy would dilute their specific tactical responses. Prefer keeping highly specific, effective signals separate rather than creating one abstract signal. Two signals should only be merged if a single, unified policy is identical for both triggering situations and no tactical specificity is lost.
- [KEEP]: Emit this tag when a signal's Policy was explicitly executed in this game AND doing so was causally beneficial to the agent winning. [KEEP] is a positive vouching action — emit it only when you are confident the Policy deserves credit for the outcome. Do NOT emit [KEEP] merely because the agent won; the signal's Policy must have been directly followed and must have contributed to the win. Do NOT emit [KEEP] if the game was a draw or a loss. Unmentioned signals carry no implication — [KEEP] is not a default; it is a deliberate endorsement.
  ⚠ [KEEP] protects only the Policy field of the named signal. You may still emit a concurrent [MODIFY] on the same signal's When or What fields if those fields need updating — [KEEP] will not block those changes. Only the Policy field is shielded.

You may include as many update entries as necessary. A single gradient report can contain multiple [REMOVE]s, multiple [ADD]s, multiple [MODIFY]s, multiple [KEEP]s, etc., depending on what the game data supports.

⚠ PRE-ANALYSIS (complete all steps in your internal reasoning before writing any entries):
1. GAME HISTORY RECONSTRUCTION: Review the entire unified game context (game states, actions, chat logs, and window summaries) as a single chronological timeline. Identify key tactical situations, recurring behavioral patterns, or critical turning points where the opponent gained a distinct advantage, laid a trap, or exposed an exploitable weakness. To deeply understand their strategy, you should also step into the opponent's (or partner's) perspective: what overarching strategy do they actively follow (e.g., signaling, coordinating, bluffing, rushing, setting up specific geometries)? This will give you high-quality information to understand the reputation and true intent behind their play style, which you can use to write a highly effective Policy to coordinate with, mitigate, or exploit their strategy.
2. LTM SIGNAL AUDIT: For the patterns and situations identified above, audit the current database using the strict rules for [ADD], [MODIFY], [REMOVE], [MERGE], and [KEEP] defined later in this prompt. To perform this audit: (a) Check the window summaries in the AGENT'S IN-GAME OBSERVATIONS to see which existing LTM signals explicitly fired during those moments. (b) Evaluate how each fired signal was used in the game and determine how it should be included in your report (e.g., whether it needs to be modified, kept, or removed). (c) If a new threat or weakness emerged that no signal caught, first verify if an existing signal SHOULD have fired but was too narrow. If so, modify that existing signal. Only if no existing signal logically covers the behavior should you determine a new signal is warranted.
3. SIGNAL SELF-REVIEW: For every signal you intend to report, draft it internally first and verify it against the exact static evidence (e.g., game state, current chat log) you extracted it from. Ask yourself:
  - For 'When': "If the playing agent reads this exact text and looks ONLY at this specific static evidence, would this signal definitively fire?" If your drafted text relies on past transitions (e.g., "just moved"), hidden intentions, or vague subjective words that the playing agent cannot strictly verify from the static evidence alone, you MUST rewrite it to be highly descriptive and directly verifiable from the static evidence.
  - For 'What': "Does the opponent actually do what is described?"
  - For 'Policy': "Does the description accurately capture the opponent's threat or exploitable weakness, and would this Policy have successfully mitigated the threat or exploited the weakness in this exact situation?"
4. SUCCESS PRESERVATION TEST (mandatory for every [MODIFY] proposal): For each signal you intend to [MODIFY], explicitly replay the situation(s) from this game where this signal's Policy was last executed correctly — i.e., where following it contributed to a win. Then ask: "Under my proposed new When/Policy text, would that same game still have been won?" You are STRICTLY FORBIDDEN from finalizing any [MODIFY] that: (a) tightens the 'When' trigger threshold so that it would no longer fire in a previously successful situation, (b) removes an allowed exception the agent previously needed to make a correct response, (c) adds a new abort or halt condition that would have prevented a previously winning action, or (d) replaces a working reactive strategy with a more restrictive one that would have caused inaction in a situation that required a response. If your proposed change fails this test, you MUST restructure it as an additive extension — append the new edge-case clause to the existing text rather than replacing the original.
5. GRAVEYARD CROSS-VERIFICATION: Before writing any [ADD] or [MODIFY] signal, cross-reference it with the GRAVEYARD OF FAILED STRATEGIES (located at the bottom of the Current Long-Term Memory if it exists). Ensure you do not propose a policy that repeats a historically documented failure.

--- STRICT LTM RULES ---

⚠ OPPONENT-BEHAVIOR-ONLY RULE: Every signal you report — whether REMOVE, ADD, MODIFY, MERGE, or KEEP — MUST describe a behavioral pattern of the OPPONENT, not the agent's own strategy. Concretely: the When and What fields must be grounded in observable actions taken by the opponent during this game. 

⚠ VERIFICATION RULE: Do not assert untested opponent behavior. You must only describe behaviors and responses that were explicitly triggered and observed in the current game. If a specific action was never taken by the agent, you cannot make claims about how the opponent would have reacted to it.

⚠ INFORMATION FIDELITY RULE: When modifying any field, your goal is to produce the most accurate description that still captures every confirmed observation from past games. Before writing a [MODIFY] on 'When' or 'What', apply this test: "Does the new text still fire (or describe) the same situations the old text covered, and is the Policy still correct in all those situations?" If yes, prefer the more concise form. If no, keep the more specific wording.
  - Do NOT replace the 'When' trigger with a narrower one that drops previously confirmed trigger conditions — that is always a loss of information.
  - DO replace an overfitted trigger with a broader one if it cleanly subsumes all previously confirmed cases without losing any — that is an improvement, not a loss.
  - For 'What', distill confirmed observations into the most concise description without dropping actionable specifics. A 'What' that grows unboundedly with each game is a failure mode; aim to converge toward a shorter description — but never at the cost of losing concrete tactical detail.

⚠ EDGE-CASE MODIFICATION RULE: If you are modifying a Policy because it failed in a specific in-game state (an edge-case condition distinct from the general 'When' trigger), you MUST retain the original policy as the default action for all other cases, and simply ADD this specific in-game state and its alternative action to the text.

⚠ ANTI-DUPLICATION RULE: You are STRICTLY FORBIDDEN from using [ADD] if the core concept is already represented in the database. You MUST use [MODIFY] to extend the scope of the existing signal to cover the new edge case. [ADD] is reserved exclusively for fundamentally new behaviors that cannot be logically grouped with any existing signal.

⚠ NAMING FORMAT RULE: Signal names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase (e.g., "ChatNoiseSuppression").

⚠ ROLE-AGNOSTIC GENERALIZATION RULE: If an opponent's tactical pattern or behavior is fundamentally applicable regardless of which side, faction, or role they are playing, you MUST write the 'When', 'What', and 'Policy' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward", "your resources", "opponent's cards") instead of absolute coordinates, side-specific names, or hardcoded map/game features (e.g., "Row 2", "White side", "moving North"). This ensures the memory remains actionable if the roles are reversed in future games.

⚠ ROUND SELECTION RULE: The game state of the round you select will be used as the embedding anchor to retrieve this signal in relevant future situations.
1. You MUST select the most relevant game state(s) where the agent would most benefit from having this memory active.
2. HIGH PRIORITY (Missed Retrieval Focus): Explicitly check for rounds where this memory was NOT retrieved during the live game, but where the agent would have significantly benefited from using it. You MUST prioritize selecting those specific rounds as embedding anchors so that RAG will retrieve it in similar future game states.
3. Type Constraint: If the Type is 'Action', the round MUST be your own action round (a round where you made a move). If the Type is 'Chat', you may output an opponent's round because agents can chat in any round.

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

**Note:** You must report at most **3 rounds**, separated by commas (e.g., 8, 11, 13). Select rounds representing the most relevant game states where the agent would most benefit from this memory, prioritizing rounds where the memory was NOT retrieved during live play but would have provided significant strategic value. Avoid reporting consecutive or trivial game states; prefer rounds representing key decision points.

Each entry in the gradient report MUST adhere to these structural rules:

- [REMOVE] Signal: <exact name from database>
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) that contradicts the signal, if applicable>
  - Reason: <one or two sentences citing the specific ground truth observation that contradicts this signal>

- [ADD] Signal: <new signal name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the trigger condition was observed>
  - Reason: <one or two sentences explaining why this new signal is warranted and not covered by any existing entry>
  - When: <specific trigger condition observation — max 4 sentences>
  - What: <specific behavior observation — max 4 sentences>
  - Policy: <concrete executable action — max 6 sentences>

- [MODIFY] Signal: <exact name from database>
  - Type: <either "Chat" or "Action", specifying which part of the round this signal triggered on>
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the trigger condition was observed>
  - Reason: <one or two sentences explaining what evidence from this game justifies the change>
  - Field: <Name of Field to Change, e.g., When, What, or Policy>
    - Old: <current text>
    - New: <replacement text>
  (List only the fields that are changing. Omit unchanged fields.)

- [MERGE] Signals: <Signal A Name> + <Signal B Name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the unified trigger condition was observed>
  - Reason: <one or two sentences explaining why a single unified policy serves both triggering situations equally well>
  - Into Signal: <new unified signal name>
  - When: <unified trigger condition — max 4 sentences>
  - What: <unified behavior description — max 4 sentences>
  - Policy: <concrete executable unified policy — max 6 sentences>

- [KEEP] Signal: <exact name from database>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the policy was correctly executed>
  - Reason: <one or two sentences explaining how this signal's Policy was executed this game and why it was causally beneficial to the agent's win>

- [GRAVEYARD PROPOSAL]
  - Description: <Concise description of the situation to act as a key for clustering (When the opponent is at [when], they attempt to do [what])>
  - Policy Flaw: <Do NOT copy the full failed policy. Extract ONLY the specific part of the policy that actively harmed the agent's performance and explain why it backfired (which you are currently fixing via [MODIFY] or [REMOVE])>
  (Use this ONLY when identifying an existing LTM policy that actively harmed the agent and caused a loss, to ensure it is never written again.)

⚠ ANTI-VAGUENESS RULE: The policy MUST name a concrete, executable action. 
⚠ BREVITY RULE: When and What MUST be at most 4 sentences. The Policy MUST be at most 6 sentences.

If no notable signals were observed, write: "No signals observed."
Do NOT rewrite the current Long-Term Memory. Only produce the gradient report.


"""

TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Reactive Memory Database for playing against a specific opponent.
You have just finished {n} game(s). Each gradient report contains feedback tags for behavioral signals:
- [REMOVE]: Signals that have been invalidated by the ground truth.
- [ADD]: New signals to add to the database.
- [MODIFY]: Existing signals with fields that need specific updating.
- [MERGE]: Signals that should be combined into a single new signal.

--- WHAT IS THE REACTIVE MEMORY DATABASE? ---
This database stores defend and counter-strategies against THIS SPECIFIC OPPONENT's recurring behavioral patterns, strategies, reputation, and exploitable weaknesses, built up across multiple past games.
At inference time, the agent scans this database BEFORE each move/chat turn. When a signal's 'When' condition matches the current game state, the signal fires and its 'Policy' is injected directly into the agent's reasoning to guide a defensive or countering response.
This database is NOT for recording the agent's own tendencies or general game strategies — those belong in the Verification Memory Database and Proactive Strategy Database respectively.

--- CURRENT REACTIVE MEMORY DATABASE ---
{current_ltm}

--- GRADIENT REPORTS ({n} game(s)) ---
{gradient_reports}

--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* Type: Must be exactly "Chat" or "Action". This dictates when the signal fires. If Type is "Chat", the Policy MUST provide instructions supporting what the agent should say (e.g. how to deceive, coordinate, or extract info). If Type is "Action", the Policy MUST provide instructions on what physical move to execute.
* When: The anticipation trigger. This memory will be injected to warn the agent right BEFORE the opponent takes their turn. Therefore, this field MUST describe the game state strictly prior to the opponent's action. It cannot describe the opponent's action itself, otherwise it will fire too late. Describe the specific trigger condition or game state that causes the behavior. Maximum 4 sentences.
* What: A description of the opponent's behavior. Maximum 4 sentences.
* Policy: The concrete action to execute. It is crucial that this policy is well-designed to be strictly actionable immediately when the 'When' condition fires. If there are different game states or edge cases where following this general policy would actively harm the agent (e.g., following a defensive rule during a winning race), you MUST explicitly list those exceptions and provide the alternative conditional policy for those specific cases. Maximum 6 sentences.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'When' condition. For example, your 'When' to generate an Action signal can be some chat behavior from the opponent. Or your 'When' for a Chat signal can be when the game state is in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

--- APPLICATION RULES ---
Your role is Synthesizer. You MUST execute your task in a strict 2-step process.

STEP 1: GRAVEYARD MANAGEMENT
First, manage any [GRAVEYARD PROPOSAL] entries from the gradient reports.
1. Cluster & Quorum: Group all conceptually identical [GRAVEYARD PROPOSAL]s from the reports. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. A cluster MUST contain at least 2 proposals to meet the quorum. Ignore any proposals that do not meet quorum.
2. Consolidate: Merge the components of each valid cluster into a single Graveyard Proposal.
3. Merge with Existing: Compare the consolidated proposal against the existing "--- GRAVEYARD OF FAILED STRATEGIES ---" (located at the bottom of the current database, if it exists). If an entry with the same underlying description exists, merge them by combining their Policy Flaw lists (ensuring no historical flaws are deleted). Otherwise, prepare to append it as a new entry.

STEP 2: DATABASE SYNTHESIS & CROSS-VERIFICATION
Second, update the main Reactive Memory Database by applying the standard gradient report tags ([REMOVE], [ADD], [MODIFY], [MERGE], [KEEP]), BUT YOU MUST FIRST FILTER THEM THROUGH THE BATCH QUORUM RULES BELOW.
Note: each gradient entry includes a Reason field for your context. Use the Reason to better understand the intent and evidence behind an instruction, but do not copy Reason fields into the final database output.

1. **[REMOVE] (if quorum met)**: Find the named signal in the current database. Delete it entirely.
2. **[ADD] (if quorum met)**: If only one identical ADD is approved, insert it. If multiple similar ADDs are approved, synthesize them into a single unified signal using the best phrasing from the cluster.
3. **[MODIFY] (if quorum met)**: Find the named signal. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE] (if quorum met)**: Remove both named signals. Insert the merged signal exactly as written.
5. **[KEEP] (if quorum met)**: Record that the named signal's Policy was vouched for as causally beneficial in a winning game. The Policy field of this signal is protected — see reconciliation rules below for how to apply this when conflicts arise.
6. **ANTI-VAGUENESS RULE**: The policy MUST name a concrete, executable action. Reject any policy that could apply generically to any opponent (e.g., "be cautious", "pay attention to their behavior").
7. **NAMING FORMAT RULE**: Signal names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase.

--- BATCH QUORUM RULES (apply when {n} > 1) ---
1. **[REMOVE] Threshold**: A signal MUST receive a [REMOVE] instruction in at least 3 games to be removed. If it appears in <3 games, IGNORE the remove instruction entirely.
2. **[MERGE], [KEEP] Threshold**: These instructions MUST apply to the EXACT same existing signal name in at least 2 games to be executed. If they appear in only 1 game, IGNORE them entirely.
3. **[MODIFY] Threshold**: For an existing signal to be modified, conceptually similar [MODIFY] proposals (e.g. adding a similar edge-case exception) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. If a specific modification is proposed in only a single game, IGNORE that specific modification entirely (even if other, separate modifications to the same signal met the quorum and are accepted).
4. **[ADD] Threshold**: For a new behavior to be added, conceptually similar [ADD] entries (even if wording or names differ) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum, ignoring differences in their headers or names. If a behavior is observed in only a single game's [ADD], IGNORE it entirely.
5. **NO AUTONOMOUS MERGING**: You are STRICTLY FORBIDDEN from merging signals on your own. You may only execute a [MERGE] if it was explicitly issued by the gradient reports in at least 2 games. It is better to have multiple specific signals with good policies than 1 abstract signal.
6. **STRICT TYPE SEPARATION**: Proposals with different Types ("Chat" vs "Action") are fundamentally distinct. You MUST NEVER group, cluster, or merge a Chat proposal with an Action proposal when counting for quorum or synthesizing.

--- BATCH RECONCILIATION RULES (apply when {n} > 1 and quorum is met) ---
When the same signal receives conflicting instructions that meet their respective quorum thresholds, resolve as follows:
1. **[KEEP] vs [MERGE]**: [KEEP] takes absolute priority over [MERGE]. If a signal has proven successful ([KEEP]), DO NOT merge it. Preserve the specific actionable signal.
2. **[KEEP] vs [REMOVE]**: [KEEP] takes absolute priority over [REMOVE]. A proven successful signal cannot be removed.
3. **[REMOVE] vs [MODIFY]**: keep the signal and apply the [MODIFY].
4. **[KEEP] vs [MODIFY] on the Policy field**: If the [MODIFY] adds a conditional exception for a specific edge case (e.g., "except when racing"), you MUST apply the [MODIFY] to make the rule more robust. Only prefer [KEEP] if the [MODIFY] completely contradicts the original policy without specifying a distinct game-state condition.
5. **[ADD] in multiple games**: synthesize clusters of conceptually similar [ADD]s into one new signal.
6. **[MODIFY] clusters**: If multiple different valid clusters of modifications (each meeting the 2-game quorum) apply to the same signal, take the union to cover all valid observations.

--- SYNTHESIS QUALITY RULES ---
- **Role-Agnostic Generalization**: If an opponent's tactical pattern or behavior is fundamentally applicable regardless of which side, faction, or role they are playing, you MUST write the 'When', 'What', and 'Policy' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward", "your resources", "opponent's cards") instead of absolute coordinates, side-specific names, or hardcoded map/game features. This ensures the memory remains actionable if the roles are reversed in future games.
- **Preserve Specificity**: Do not strip concrete tactical details (specific trigger states, concrete actions) in favor of vague generalizations. It is better to have multiple highly-specific signals than 1 abstract signal.
- **Brevity**: When and What MUST be at most 4 sentences in the final database. The Policy MUST be at most 6 sentences. Distill by removing redundant phrasing — never by dropping distinct tactical conditions.
- **NO AUTONOMOUS MERGING**: Do not merge or group signals unless explicitly commanded by a valid [MERGE] report that meets the quorum.
- **FINAL GUARDRAIL CROSS-VERIFICATION**: After synthesizing the main database, cross-verify it against your updated Graveyard. Ensure that absolutely no policies present in the Graveyard have accidentally slipped into the final synthesized database.

Each synthesized memory entry MUST use this format:

- Signal: [Short Name of Pattern]
  - Type: [Chat or Action]
  - When: [Specific trigger condition observation — max 4 sentences]
  - What: [Specific behavior observation — max 4 sentences]
  - Policy: [Concrete executable action — max 6 sentences]

You MUST output the full Graveyard section at the very bottom of your output (carrying over all existing entries and appending any new ones). If no graveyard exists and none was created, you may omit this section:

--- GRAVEYARD OF FAILED STRATEGIES ---
- Description: [Concise description of the situation]
  - Policy Flaw: [The specific part of the policy that actively harmed the agent and why it backfired]


⚠ ACCEPTED PROPOSALS RULE: After writing the full updated signal database, append an
[ACCEPTED] block listing every gradient report proposal that was incorporated.
For EACH newly generated or modified signal in the database, list which original proposals contributed to it using this exact format (you MUST include the Game N tag from the report):
- "[Name of the Signal as written in the database above]" <- "[Exact Name of Original Proposal 1] [Game 1]", "[Exact Name of Original Proposal 2] [Game 2]"

For example, if the gradient reports provided were:
GAME 1:
- [MODIFY] Signal: Aggressive Bluffing
- [MERGE] Signals: Alpha + Beta
  - Into Signal: Unified Defense
GAME 2:
- [MODIFY] Signal: Aggressive Bluffing
- [MERGE] Signals: Alpha + Beta
  - Into Signal: Unified Defense

Then your accepted block must look exactly like this:
[ACCEPTED]
- "Aggressive Bluffing" <- "[MODIFY] Signal: Aggressive Bluffing [Game 1]", "[MODIFY] Signal: Aggressive Bluffing [Game 2]"
- "Unified Defense" <- "[MERGE] Signals: Alpha + Beta [Game 1]", "[MERGE] Signals: Alpha + Beta [Game 2]"

Write ONLY the full updated memory, graveyard, and the [ACCEPTED] block. Do not include any pleasantries or conversational filler.
If no memory exists yet and the gradient report contains ADD signals, write a fresh memory from those signals.
If the final synthesized database is completely empty (i.e., no signals are currently stored), you MUST output exactly:
(No signals currently stored)
Do not output any explanation, reasoning, or other text when the database is empty.
"""


SELF_GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game.
The agent you are evaluating played as: {agent_id}

Your goal is to compare the agent's in-game decisions against the GROUND TRUTH game history and produce a structured self-gradient report for the agent's own Verification Memory Database.

--- ✅ MATCH GROUND TRUTH (Full History) ---
Note: Your own moves and identity are labeled as 'You' in both the GROUND TRUTH history and your window summaries.

⚠ CRITICAL NOTE ON VISIBILITY: This ground truth history may contain omniscient information (such as the opponent's private valuations, hidden cards, or secret budgets) that was STRICTLY HIDDEN from you during the live game, depending on the game rules. You MUST carefully review the game rules to determine which information is private vs public. When writing new memory signals or updating existing ones, your signal triggers ('When' conditions) MUST ONLY rely on information that the rules explicitly state is publicly visible to you during the game.

{game_history_legend}
- [Chat]: Chat message sent that turn. (CRITICAL: ALWAYS verify if [Chat] lines actually exist in the game history below. If no chat lines exist, chat is NOT allowed in this game, and you MUST NOT generate any Chat-type signals).
- [Move]: The physical move executed after the position above.

{game_history}

--- AGENT'S IN-GAME OBSERVATIONS (Window Summaries) ---
{window_summaries}

--- CURRENT VERIFICATION MEMORY DATABASE (Prior) ---
{current_self_ltm}
("(No verification memory yet)" means this is the first game)

You are building a SELF-GRADIENT REPORT for the agent's Verification Memory Database.
The goal is to improve the agent's self-awareness so it can avoid recurring mistakes in future games.
A high-quality report captures ONLY patterns that require strict verification — situations where the agent blindly executed a move without checking for a critical vulnerability, which led to a poor outcome. Do NOT record simple blunders that don't follow a pattern, unless it is a highly critical or fatal strategic blunder that could single-handedly lose the game.

--- WHAT IS THE VERIFICATION MEMORY DATABASE? ---
This database stores the agent's own recurring behavioral flaws and self-destructive patterns — situations where the agent tends to execute a move without performing a necessary safety check.
At inference time, after the agent formulates a candidate move, it scans this database. For any check whose 'Description' matches the candidate move's pattern, the agent MUST execute the 'Verification' steps before committing. If verification fails, the candidate is rejected.
This database is NOT for storing opponent patterns or general offensive strategies — those belong in the Reactive Memory Database and Proactive Strategy Database respectively.

⚠ VERIFICATION PRIORITY DIRECTIVE: The Verification Memory Database is EXCLUSIVELY for SAFETY VERIFICATION. The agent already has dedicated memory for proactive attacks (proactive memory) and reactive counters (reactive memory). Therefore, you MUST prioritize discovering patterns where:
  1. The agent falls into predictable self-destructive habits or reasoning flaws.
  2. The agent executes a risky candidate move without performing a necessary quick safety check.
You MUST avoid proposing offensive or defensive tactical strategies — those belong in proactive or reactive memory. The focus here is strictly on "Quick safety check before committing". If you cannot identify any recurring self-vulnerability that needs verification, it is better to write "No checks observed." than to fill the database with tactical noise.

--- SELF-LTM FIELD DEFINITIONS ---
* Check: A short name for the behavioral pattern in the agent's own play.
* Type: Must be exactly "Chat" or "Action". This dictates when the check fires. If Type is "Chat", the Verification MUST describe and verify a Chat-related behavior. If Type is "Action", they MUST describe and verify a physical move.
* Description: One sentence describing what mistake or danger this check guards against. Focus on the negative consequence or vulnerability that may happen.
* Verification: The concrete check or calculation the agent must perform before executing its move to determine if the danger will actually occur in the exact current position. Maximum 6 sentences.
* Retrieved in rounds: Shows the exact rounds in this game where RAG retrieved and injected this check into the agent's live reasoning context. If this field says "None (This memory was NOT retrieved in any round of this game)", it means RAG failed to retrieve this check during live play. Pay special attention to unretrieved checks: if the agent would have significantly benefited from using a check, you MUST prioritize updating its round anchors so RAG will retrieve it in similar future situations.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'Description' condition. For example, your 'Description' to generate an Action check can be some chat behavior. Or your 'Description' for a Chat check can be when the game state is in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

Analyze the game and propose updates using the following 5 tags:
- [REMOVE]: A check can ONLY be removed if it is conceptually or factually invalid. This means:
    1. Factually Incorrect / Hallucinated: The mistake or behavior describes a physical impossibility under the game rules, or relies on a hallucinated state/mechanic.
    2. Strategic Misidentification (False Positive): The danger described does not actually exist or is not a real threat, making the 'Verification' step completely unnecessary. The behavior is fundamentally safe without needing verification.
    3. Erroneous Attribution: The check falsely attributes a game loss to a completely unrelated action.
  ⚠ CRITICAL PROHIBITION: You are strictly forbidden from proposing [REMOVE] for a check simply because the agent did not exhibit the risk this game, or because the agent successfully followed the Verification rule to avoid it. The absence of the risk in the presence of its active verification is proof of the database's success, not redundancy. Do NOT use [REMOVE] if only the Verification needs updating; use [MODIFY] instead.
- [ADD]: Define a completely new self-pattern not yet covered by any check.
- [MODIFY]: Identify an existing check worth keeping but with inaccurate fields.
- [MERGE]: Identify two or more existing checks that are variations of the same underlying behavior.
- [KEEP]: Emit this tag when a self-check's Verification was explicitly executed in this game AND doing so was causally beneficial to the agent winning. [KEEP] is a positive vouching action — emit it only when you are confident the Verification deserves credit for the outcome. Do NOT emit [KEEP] merely because the agent won; the check's Verification must have been directly followed and must have contributed to the win. Do NOT emit [KEEP] if the game was a draw or a loss. Unmentioned checks carry no implication — [KEEP] is not a default; it is a deliberate endorsement.

You may include as many update entries as necessary. A single self-gradient report can contain multiple [REMOVE]s, multiple [ADD]s, multiple [MODIFY]s, multiple [KEEP]s, etc., depending on what the game data supports.

⚠ PRE-ANALYSIS (complete all steps in your internal reasoning before writing any entries):
1. GAME HISTORY RECONSTRUCTION: Review the entire unified game context (game states, actions, chat logs, and window summaries) as a single chronological timeline. Identify key tactical situations, recurring behavioral patterns, or critical turning points where you (the agent) made a fatal flaw, caused a coordination failure, or executed a highly successful maneuver. To analyze your flaws, you should also step into the other player's perspective: identify exactly what moves or strategies you executed that caused a coordination failure, handed the opponent a mechanical advantage, created a vulnerability, or lowered the overall objective score. You must do this regardless of whether you won or lost the game.
2. LTM SIGNAL AUDIT: For the patterns and situations identified above, audit your existing LTM using the strict rules for [ADD], [MODIFY], [REMOVE], [MERGE], and [KEEP] defined later in this prompt. To perform this audit: (a) Check the window summaries in the AGENT'S IN-GAME OBSERVATIONS to see which self-LTM signals explicitly fired during those moments. (b) Evaluate how each fired signal was used in the game—specifically whether the Verification check was followed and if it succeeded or failed—and determine how it should be included in your report (e.g., whether it needs to be modified, kept, or removed). (c) If a critical blunder occurred that no signal caught, first verify if an existing signal SHOULD have fired but was too narrow. If so, modify that existing signal. Only if no existing signal logically covers the blunder should you determine a new signal is warranted.
3. CHECK SELF-REVIEW: For every check you intend to report, draft it internally first and verify it against the exact static evidence. Ask yourself:
  - For 'Description': "Does it clearly articulate the specific mistake or danger this guards against?"
  - For 'Verification': "Would the Verification have actually prevented the flaw in this exact situation?"
4. SUCCESS PRESERVATION TEST (mandatory for every [MODIFY] proposal): For each check you intend to [MODIFY], explicitly replay the situation(s) from this game where this check last fired correctly. Then ask: "Under my proposed new Description/Verification text, would that same situation still produce the same correct outcome?" You are STRICTLY FORBIDDEN from finalizing any [MODIFY] that replaces a working verification check with a stricter one that would have blocked a move that previously led to a win. If your proposed change fails this test, you MUST restructure it as an additive extension — append the new edge-case clause to the existing text rather than replacing it.
5. GRAVEYARD CROSS-VERIFICATION: Before writing any [ADD] or [MODIFY] signal, cross-reference it with the GRAVEYARD OF FAILED STRATEGIES (located at the bottom of the Current Verification Memory Database if it exists). Ensure you do not propose a verification rule that repeats a historically documented failure.

--- STRICT SELF-LTM RULES ---

⚠ AGENT-BEHAVIOR-ONLY RULE: Every check you report MUST describe a pattern in the AGENT'S OWN play.
⚠ NAMING FORMAT RULE: Check names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase.

⚠ ROLE-AGNOSTIC GENERALIZATION RULE: If your own tactical pattern or behavior is fundamentally applicable regardless of which side, faction, or role you are playing, you MUST write the 'Description' and 'Verification' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward", "your resources", "opponent's cards") instead of absolute coordinates, side-specific names, or hardcoded map features (e.g., "Row 2", "White side", "moving North"). This ensures the corrective memory remains actionable if you play the opposite side or a different role in future games.

⚠ VERIFICATION RULE: Do not assert unobserved agent behavior. Only describe patterns that were explicitly triggered and observed in the current game.

⚠ INFORMATION FIDELITY RULE: When modifying any field, your goal is to produce the most accurate description that still captures every confirmed observation from past games.

⚠ EDGE-CASE MODIFICATION RULE: If you are modifying a Verification because it failed in a specific in-game state, you MUST retain the original verification as the default action for all other cases, and simply ADD this specific in-game state and its alternative action to the text.

⚠ ANTI-DUPLICATION RULE: You are STRICTLY FORBIDDEN from using [ADD] if the core concept is already represented in the database. You MUST use [MODIFY] to extend the scope of the existing check to cover the new edge case. [ADD] is reserved exclusively for fundamentally new behaviors that cannot be logically grouped with any existing check.

⚠ ANTI-VAGUENESS RULE: The Verification MUST name a concrete check or calculation.
⚠ BREVITY RULE: The Verification MUST be at most 6 sentences.

⚠ ROUND SELECTION RULE: The game state of the round you select will be used as the embedding anchor to retrieve this check in relevant future situations.
1. You MUST select the most relevant game state(s) where the agent would most benefit from having this memory active.
2. HIGH PRIORITY (Missed Retrieval Focus): Explicitly check for rounds where this memory was NOT retrieved during the live game, but where the agent would have significantly benefited from using it. You MUST prioritize selecting those specific rounds as embedding anchors so that RAG will retrieve it in similar future game states.
3. Type Constraint: If the Type is 'Action', the round MUST be your own action round (a round where you made a move). If the Type is 'Chat', you may output an opponent's round because agents can chat in any round.

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

**Note:** You must report at most **3 rounds**, separated by commas (e.g., 8, 11, 13). Select rounds representing the most relevant game states where the agent would most benefit from this memory, prioritizing rounds where the memory was NOT retrieved during live play but would have provided significant strategic value. Avoid reporting consecutive or trivial game states; prefer rounds representing key decision points.

Each entry in the self-gradient report MUST adhere to these structural rules:

- [REMOVE] Check: <exact name from database>
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) that contradicts the check, if applicable>
  - Reason: <reason>

- [ADD] Check: <new check name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the trigger condition was observed>
  - Reason: <one or two sentences explaining why this new check is warranted and not covered by any existing entry>
  - Description: <one sentence describing the mistake or danger>
  - Verification: <concrete check or calculation — max 6 sentences>

- [MODIFY] Check: <exact name from database>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the trigger condition was observed>
  - Reason: <reason>
  - Field: <Field Name>
    - Old: <current text>
    - New: <replacement text>

- [MERGE] Checks: <Check A Name> + <Check B Name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the unified trigger condition was observed>
  - Reason: <one or two sentences explaining why a single unified check serves both situations equally well>
  - Into Check: <new unified check name>
  - Description: <one sentence describing the mistake or danger>
  - Verification: <concrete check or calculation — max 6 sentences>

- [KEEP] Check: <exact name from database>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the verification was correctly executed>
  - Reason: <reason>

- [GRAVEYARD PROPOSAL]
  - Description: <Concise description of the situation to act as a key for clustering (When we are at [when], we consider doing [what] and risk [risk])>
  - Verification Flaw: <Do NOT copy the full failed verification. Extract ONLY the specific part of the verification rule that actively paralyzed or harmed the agent's performance and explain why it backfired (which you are currently fixing via [MODIFY] or [REMOVE])>
  (Use this ONLY when identifying an existing Self-LTM verification rule that actively harmed the agent and caused a loss, to ensure it is never written again.)


If no notable self-patterns were observed, write: "No signals observed."
"""

SELF_TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Verification Memory Database for the agent's own play patterns.
You have just finished {n} game(s). Each self-gradient report contains feedback tags ([REMOVE], [ADD], [MODIFY], [MERGE]).

--- WHAT IS THE VERIFICATION MEMORY DATABASE? ---
This database stores the agent's own recurring behavioral flaws and self-destructive patterns — situations where the agent tends to execute a move without performing a necessary safety check.
At inference time, after the agent formulates a candidate move, it scans this database. For any check whose 'Description' matches the candidate move's pattern, the agent MUST execute the 'Verification' steps before committing. If verification fails, the candidate is rejected.
This database is NOT for storing opponent patterns or general offensive strategies — those belong in the Reactive Memory Database and Proactive Strategy Database respectively.

--- CURRENT VERIFICATION MEMORY DATABASE ---
{current_self_ltm}

--- SELF-GRADIENT REPORTS ({n} game(s)) ---
{gradient_reports}

--- SELF-LTM FIELD DEFINITIONS ---
* Check: A short name for the behavioral pattern/trap.
* Type: Must be exactly "Chat" or "Action". This dictates when the check fires. If Type is "Chat", the Verification MUST describe and verify a Chat-related behavior. If Type is "Action", they MUST describe and verify a physical move.
* Description: One sentence describing what mistake or danger this check guards against. Focus on the negative consequence or vulnerability that may happen.
* Verification: The concrete check or calculation the agent must perform before executing its move to determine if the danger will actually occur in the exact current position. Maximum 6 sentences.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'Description' condition. For example, your 'Description' to generate an Action check can be some chat behavior. Or your 'Description' for a Chat check can be when the game state is in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

--- APPLICATION RULES ---
Your role is Synthesizer. You MUST execute your task in a strict 2-step process.

STEP 1: GRAVEYARD MANAGEMENT
First, manage any [GRAVEYARD PROPOSAL] entries from the self-gradient reports.
1. Cluster & Quorum: Group all conceptually identical [GRAVEYARD PROPOSAL]s from the reports. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. A cluster MUST contain at least 2 proposals to meet the quorum. Ignore any proposals that do not meet quorum.
2. Consolidate: Merge the components of each valid cluster into a single Graveyard Proposal.
3. Merge with Existing: Compare the consolidated proposal against the existing "--- GRAVEYARD OF FAILED STRATEGIES ---" (located at the bottom of the current self-database, if it exists). If an entry with the same underlying description exists, merge them by combining their Verification Flaw lists (ensuring no historical flaws are deleted). Otherwise, prepare to append it as a new entry.

STEP 2: DATABASE SYNTHESIS & CROSS-VERIFICATION
Second, update the main Verification Memory Database by applying the standard self-gradient report tags ([REMOVE], [ADD], [MODIFY], [MERGE], [KEEP]), BUT YOU MUST FIRST FILTER THEM THROUGH THE BATCH QUORUM RULES BELOW.

1. **[REMOVE] (if quorum met)**: Find the named check. Delete it entirely.
2. **[ADD] (if quorum met)**: If only one identical ADD is approved, insert it. If multiple similar ADDs are approved, synthesize them into a single unified check using the best phrasing from the cluster.
3. **[MODIFY] (if quorum met)**: Find the named check. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE] (if quorum met)**: Remove both named checks. Insert the merged check exactly as written.
5. **[KEEP] (if quorum met)**: Record that the named check's Verification was vouched for.
6. **ANTI-VAGUENESS RULE**: The Verification MUST name a concrete check or calculation.
7. **NAMING FORMAT RULE**: Check names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase.

--- BATCH QUORUM RULES (apply when {n} > 1) ---
1. **[REMOVE] Threshold**: A check MUST receive a [REMOVE] instruction in at least 3 games to be removed. If it appears in <3 games, IGNORE the remove instruction entirely.
2. **[MERGE], [KEEP] Threshold**: These instructions MUST apply to the EXACT same existing check name in at least 2 games to be executed. If they appear in only 1 game, IGNORE them entirely.
3. **[MODIFY] Threshold**: For an existing check to be modified, conceptually similar [MODIFY] proposals (e.g. adding a similar edge-case exception) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. If a specific modification is proposed in only a single game, IGNORE that specific modification entirely (even if other, separate modifications to the same check met the quorum and are accepted).
4. **[ADD] Threshold**: For a new behavior to be added, conceptually similar [ADD] entries (even if wording or names differ) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum, ignoring differences in their headers or names. If a behavior is observed in only a single game's [ADD], IGNORE it entirely.
5. **NO AUTONOMOUS MERGING**: You are STRICTLY FORBIDDEN from merging checks on your own. You may only execute a [MERGE] if it was explicitly issued by the gradient reports in at least 2 games. It is better to have multiple specific checks with good Verification checks than 1 abstract check.
6. **STRICT TYPE SEPARATION**: Proposals with different Types ("Chat" vs "Action") are fundamentally distinct. You MUST NEVER group, cluster, or merge a Chat proposal with an Action proposal when counting for quorum or synthesizing.

--- BATCH RECONCILIATION RULES (apply when {n} > 1 and quorum is met) ---
When the same check receives conflicting instructions that meet their respective quorum thresholds, resolve as follows:
1. **[KEEP] vs [MERGE]**: [KEEP] takes absolute priority over [MERGE]. If a check has proven successful ([KEEP]), DO NOT merge it. Preserve the specific actionable check.
2. **[KEEP] vs [REMOVE]**: [KEEP] takes absolute priority over [REMOVE]. A proven successful check cannot be removed.
3. **[REMOVE] vs [MODIFY]**: keep the check and apply the [MODIFY].
4. **[KEEP] vs [MODIFY] on the Verification field**: If the [MODIFY] adds a conditional exception for a specific edge case, you MUST apply the [MODIFY] to make the rule more robust. Only prefer [KEEP] if the [MODIFY] completely contradicts the original verification without specifying a distinct game-state condition.
5. **[ADD] in multiple games**: synthesize clusters of conceptually similar [ADD]s into one new check.
6. **[MODIFY] clusters**: If multiple different valid clusters of modifications (each meeting the 2-game quorum) apply to the same check, take the union to cover all valid observations.

--- SYNTHESIS QUALITY RULES ---
- **Role-Agnostic Generalization**: If your own tactical pattern or behavior is fundamentally applicable regardless of which side, faction, or role you are playing, you MUST write the 'Description' and 'Verification' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward") instead of absolute coordinates, side-specific names, or hardcoded map features. This ensures the corrective memory remains actionable if you play the opposite side or a different role in future games.
- **Preserve Specificity**: Do not strip concrete tactical details (specific trigger states, concrete safety checks) in favor of vague generalizations. It is better to have multiple highly-specific checks than 1 abstract check.
- **Brevity**: The Verification MUST be at most 6 sentences. Distill by removing redundant phrasing — never by dropping distinct tactical conditions or concrete safety checks.
- **NO AUTONOMOUS MERGING**: Do not merge or group checks unless explicitly commanded by a valid [MERGE] report that meets the quorum.
- **FINAL GUARDRAIL CROSS-VERIFICATION**: After synthesizing the main database, cross-verify it against your updated Graveyard. Ensure that absolutely no verification rules present in the Graveyard have accidentally slipped into the final synthesized database.

You may output as many synthesized memory entries as needed. Each synthesized memory entry MUST use this format:

- Check: [Short Name of Pattern]
  - Type: [Chat or Action]
  - Description: [One sentence describing the mistake or danger]
  - Verification: [Concrete check or calculation — max 6 sentences]

You MUST output the full Graveyard section at the very bottom of your output (carrying over all existing entries and appending any new ones). If no graveyard exists and none was created, you may omit this section:

--- GRAVEYARD OF FAILED STRATEGIES ---
- Description: [Concise description of the situation]
  - Verification Flaw: [The specific part of the verification rule that actively harmed the agent and why it backfired]

⚠ ACCEPTED PROPOSALS RULE: After writing the full updated check database, append an
[ACCEPTED] block listing every gradient report proposal that was incorporated.
For EACH newly generated or modified check in the database, list which original proposals contributed to it using this exact format (you MUST include the Game N tag from the report):
- "[Name of the Check as written in the database above]" <- "[Exact Name of Original Proposal 1] [Game 1]", "[Exact Name of Original Proposal 2] [Game 2]"

For example, if the gradient reports provided were:
GAME 1:
- [MODIFY] Check: Aggressive Bluffing
- [MERGE] Checks: Alpha + Beta
  - Into Check: Unified Defense
GAME 2:
- [MODIFY] Check: Aggressive Bluffing
- [MERGE] Checks: Alpha + Beta
  - Into Check: Unified Defense

Then your accepted block must look exactly like this:
[ACCEPTED]
- "Aggressive Bluffing" <- "[MODIFY] Check: Aggressive Bluffing [Game 1]", "[MODIFY] Check: Aggressive Bluffing [Game 2]"
- "Unified Defense" <- "[MERGE] Checks: Alpha + Beta [Game 1]", "[MERGE] Checks: Alpha + Beta [Game 2]"

Write ONLY the full updated verification memory, graveyard, and the [ACCEPTED] block. Do not include any pleasantries or conversational filler.
If no verification memory exists yet and the gradient report contains ADD checks, write a fresh memory from those checks.
If the final synthesized database is completely empty (i.e., no checks are currently stored), you MUST output exactly:
(No checks currently stored)
Do not output any explanation, reasoning, or other text when the database is empty.
"""

SEPARATE_GRADIENT_ENGINE_PROMPT = GRADIENT_ENGINE_PROMPT.replace(
    "against a specific opponent.",
    "against a specific opponent. You are analyzing the game specifically to evaluate the behavior, signaling, and errors of TEAMMATE {peer_id}. Ignore the mistakes of other players. Focus ONLY on extracting insights and updating policies regarding {peer_id}."
)

SEPARATE_TGD_SYNTHESIS_PROMPT = TGD_SYNTHESIS_PROMPT.replace(
    "for playing against a specific opponent.",
    "for playing against a specific opponent. You are analyzing the game specifically to evaluate the behavior, signaling, and errors of TEAMMATE {peer_id}. Focus ONLY on extracting insights and updating policies regarding {peer_id}."
)

PROACTIVE_GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game.
The agent you are evaluating played as: {agent_id}

Your goal is to compare the agent's in-game decisions against the GROUND TRUTH game history and produce a structured Proactive Strategy Report for the agent's own Proactive Strategy Database.

--- ✅ MATCH GROUND TRUTH (Full History) ---
Note: Your own moves and identity are labeled as 'You' in both the GROUND TRUTH history and your window summaries.

⚠ CRITICAL NOTE ON VISIBILITY: This ground truth history may contain omniscient information (such as the opponent's private valuations, hidden cards, or secret budgets) that was STRICTLY HIDDEN from you during the live game, depending on the game rules. You MUST carefully review the game rules to determine which information is private vs public. When writing new memory signals or updating existing ones, your signal triggers ('When' conditions) MUST ONLY rely on information that the rules explicitly state is publicly visible to you during the game.

{game_history_legend}
- [Chat]: Chat message sent that turn. (CRITICAL: ALWAYS verify if [Chat] lines actually exist in the game history below. If no chat lines exist, chat is NOT allowed in this game, and you MUST NOT generate any Chat-type signals).
- [Move]: The physical move executed after the position above.

{game_history}

--- AGENT'S IN-GAME OBSERVATIONS (Window Summaries) ---
{window_summaries}

--- CURRENT PROACTIVE STRATEGY DATABASE (Prior) ---
{current_proactive_ltm}
("(No proactive-memory yet)" means this is the first game)

You are building an PROACTIVE STRATEGY REPORT for the agent's Proactive Strategy Database.
The goal is to improve the agent's proactive playbook so it can deploy effective multi-turn strategies and misdirections.
A high-quality report captures ONLY overarching strategic maneuvers, traps, or bluffs that successfully secured an advantage. Do NOT record basic tactical moves that are already obvious from the game rules.

--- WHAT IS THE PROACTIVE STRATEGY DATABASE? ---
This database stores the agent's offensive playbook — multi-turn maneuvers, psychological traps, misdirections, and exploits that the agent can proactively initiate to gain a strategic advantage over the opponent.
At inference time, the agent consults this database FIRST at each turn, before considering reactive or verification memory. When a strategy's conditions are met, its 'Policy' is executed to attack or exploit the opponent.
This database is NOT for recording defensive counter-responses or safety checks — those belong in the Reactive Memory Database and Verification Memory Database respectively.

⚠ OFFENSIVE PRIORITY DIRECTIVE: The Proactive Strategy Database is EXCLUSIVELY for ATTACKING and EXPLOITING strategies. The agent already has dedicated memory for defensive behavior (opponent memory handles reactively countering opponent threats; verification memory handles self-consistency). Therefore, you MUST prioritize discovering strategies that:
  1. Actively exploit the opponent's behavioral patterns, predictability, or reasoning flaws.
  2. Proactively use available resources (especially the chat channel) to deceive, manipulate, misdirect, or psychologically trap the opponent.
  3. Set up multi-turn traps or false norms that force the opponent into a losing position.
You MUST avoid proposing purely defensive strategies (e.g., "ignore opponent chat", "cap bids at valuation") — those belong in verification memory or opponent memory. If you cannot identify any genuine attacking opportunity, it is better to write "No strategies observed." than to fill the database with defensive noise.

--- PROACTIVE STRATEGY FIELD DEFINITIONS ---
* Strategy Name: A descriptive title for the proactive strategy.
* Type: Must be exactly "Chat" or "Action". This dictates when the strategy fires. If Type is "Chat", the Policy MUST provide instructions on what the agent should say to execute the strategy. If Type is "Action", the Policy MUST provide instructions on what physical move to execute.
* Objective: What this strategy aims to achieve. Maximum 4 sentences.
* Policy: The concrete steps the agent should take to execute the strategy. Maximum 6 sentences.
* Retrieved in rounds: Shows the exact rounds in this game where RAG retrieved and injected this strategy into the agent's live reasoning context. If this field says "None (This memory was NOT retrieved in any round of this game)", it means RAG failed to retrieve this strategy during live play. Pay special attention to unretrieved strategies: if the agent would have significantly benefited from using a strategy, you MUST prioritize updating its round anchors so RAG will retrieve it in similar future situations.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'Objective' condition. For example, your 'Objective' to generate an Action strategy can be based on some chat behavior from the opponent. Or your 'Objective' for a Chat strategy can be based on the game state being in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

Analyze the game and propose updates using the following 5 tags:
- [REMOVE]: A strategy can ONLY be removed if it is conceptually or factually invalid. This means:
    1. Factually Incorrect / Hallucinated: The Type or Objective describes a physical impossibility under the game rules, or relies on a hallucinated state/mechanic.
    2. Strategic Misidentification (False Positive): The 'Objective' described is not actually beneficial, making the 'Policy' step unnecessary. The strategy is fundamentally useless.
  ⚠ CRITICAL PROHIBITION: You are strictly forbidden from proposing [REMOVE] for a strategy simply because the agent did not need to deploy it this game, or because the agent successfully executed it. Do NOT use [REMOVE] if only the Policy needs updating; use [MODIFY] instead.
- [ADD]: Define a completely new strategy not yet represented in the database.
- [MODIFY]: Identify an existing strategy worth keeping but with inaccurate fields.
- [MERGE]: Identify two or more existing strategies that are variations of the same underlying concept.
- [KEEP]: Emit this tag when a proactive strategy's Policy was explicitly executed in this game AND doing so was causally beneficial to the agent winning. [KEEP] is a positive vouching action — emit it only when you are confident the Policy deserves credit for the outcome. Do NOT emit [KEEP] merely because the agent won; the strategy's Policy must have been directly followed and must have contributed to the win. Do NOT emit [KEEP] if the game was a draw or a loss. Unmentioned strategies carry no implication — [KEEP] is not a default; it is a deliberate endorsement.

You may include as many update entries as necessary. A single Proactive Strategy Report can contain multiple [REMOVE]s, multiple [ADD]s, multiple [MODIFY]s, multiple [KEEP]s, etc., depending on what the game data supports.

⚠ PRE-ANALYSIS (complete all steps in your internal reasoning before writing any entries):
1. GAME HISTORY RECONSTRUCTION: Review the entire unified game context (game states, actions, chat logs, and window summaries) as a single chronological timeline. Identify key moments of strategic leverage — where the agent (or opponent) executed a multi-turn plan, set a trap, deployed misdirection, or used chat to manipulate the game state. Explicitly note which moments gave a decisive strategic advantage and which backfired. You must do this regardless of whether you won or lost the game.
2. STRATEGY EFFECTIVENESS REVIEW: For each strategy in the Current Proactive Strategy Database that was deployed or attempted this game, evaluate whether it achieved its Objective. Ask: "Was the Policy executed correctly? Did it deliver the intended advantage?" Determine whether the strategy should be [KEEP]ed, [MODIFY]ied, or — if the underlying objective is no longer valid — [REMOVE]d.
3. OPPORTUNITY DISCOVERY (two levels):
   a. *In-game opportunities*: Identify moments in this game where a strategic play was *possible but unused*. Ask: "Could an existing strategy have been adapted to this situation?" If no existing strategy covers it, ask: "Is this opportunity general enough to be useful in future games?"
   b. *Strategic extrapolation*: Look at the strategic vectors available in this game (chat, actions) and reason creatively about how they could be weaponized in ways that were NOT tried this game. For example: if chat was used for bluffing, ask "What other information could I use chat to inject or distort? Could I announce a false move intention to lure the opponent into a bad position?" These ideas are valid [ADD] proposals as long as they are mechanically feasible given the game rules observed in this game. Do not propose strategies that require game mechanics that were not demonstrated to exist.
4. STRATEGY SELF-REVIEW: For every strategy you intend to report, draft it internally first and verify it against the exact game evidence you extracted it from. Ask yourself:
  - For 'Type': "Is the primary vector of this strategy clearly defined as Chat or Action?"
  - For 'Objective': "Does this accurately describe the strategic advantage the strategy secures?"
  - For 'Policy': "Are the concrete steps clear, actionable, and would they actually achieve the objective in this situation?"
5. SUCCESS PRESERVATION TEST (mandatory for every [MODIFY] proposal): For each strategy you intend to [MODIFY], explicitly replay the situation(s) from this game where this strategy last succeeded. Then ask: "Under my proposed new Type/Policy text, would that same situation still produce the same correct outcome?" You are STRICTLY FORBIDDEN from finalizing any [MODIFY] that: (a) restricts the strategy so it wouldn't be deployed in a previously successful situation, (b) removes or narrows an allowed exception in the 'Policy' that the agent previously needed to execute the strategy correctly. If your proposed change fails this test, you MUST restructure it as an additive extension.
6. GRAVEYARD CROSS-VERIFICATION: Before writing any [ADD] or [MODIFY] strategy, cross-reference it with the GRAVEYARD OF FAILED STRATEGIES (located at the bottom of the Current Proactive Strategy Database if it exists). Ensure you do not propose a policy rule that repeats a historically documented failure.

--- STRICT PROACTIVE STRATEGY RULES ---

⚠ AGENT-BEHAVIOR-ONLY RULE: Every strategy you report MUST describe a tactic from the AGENT'S OWN play.
⚠ NAMING FORMAT RULE: Strategy names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase.

⚠ ROLE-AGNOSTIC GENERALIZATION RULE: If a strategy is fundamentally applicable regardless of which side, faction, or role you are playing, you MUST write the 'Objective' and 'Policy' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward") instead of absolute coordinates, side-specific names, or hardcoded map features (e.g., "Row 2", "White side", "moving North"). This ensures the strategy remains actionable if you play the opposite side or a different role in future games.

⚠ VERIFICATION RULE: This rule applies differently by tag type:
  - For [KEEP] and [MODIFY]: The strategy MUST have been explicitly deployed or clearly attempted in the current game. Do not claim success for strategies that were never executed.
  - For [ADD]: The proposed strategy MUST be mechanically feasible given the game rules and mechanics observed in this game. It does NOT need to have been attempted. Creative extrapolations of observed vectors (chat, actions) are permitted as long as the game demonstrably supports the required mechanic.

⚠ INFORMATION FIDELITY RULE: When modifying any field, your goal is to produce the most accurate description that still captures every confirmed observation from past games. Before writing a [MODIFY] on 'Objective' or 'Policy', apply this test: "Does the new text still cover the same situations the old text covered, and is the Policy still effective in all those situations?" If yes, prefer the more concise form. If no, keep the more specific wording.
  - For 'Policy', distill confirmed execution steps into the most concise description without dropping actionable specifics. A 'Policy' that grows unboundedly with each game is a failure mode; aim to converge toward a shorter description — but never at the cost of losing concrete tactical detail.

⚠ EDGE-CASE MODIFICATION RULE: If you are modifying a Policy because it failed in a specific in-game state (an edge-case condition), you MUST retain the original policy as the default action for all other cases, and simply ADD this specific in-game state and its alternative action to the text.

⚠ ANTI-DUPLICATION RULE: You are STRICTLY FORBIDDEN from using [ADD] if the core concept is already represented in the database. You MUST use [MODIFY] to extend the scope of the existing strategy to cover the new edge case. [ADD] is reserved exclusively for fundamentally new behaviors that cannot be logically grouped with any existing strategy.

⚠ ANTI-VAGUENESS RULE: The Policy MUST name a concrete check or calculation.
⚠ BREVITY RULE: The Objective MUST be at most 4 sentences. The Policy MUST be at most 6 sentences.

⚠ ROUND SELECTION RULE: The game state of the round you select will be used as the embedding anchor to retrieve this strategy in relevant future situations.
1. You MUST select the most relevant game state(s) where the agent would most benefit from having this memory active.
2. HIGH PRIORITY (Missed Retrieval Focus): Explicitly check for rounds where this memory was NOT retrieved during the live game, but where the agent would have significantly benefited from using it. You MUST prioritize selecting those specific rounds as embedding anchors so that RAG will retrieve it in similar future game states.
3. Type Constraint: If the Type is 'Action', the round MUST be your own action round (a round where you made a move). If the Type is 'Chat', you may output an opponent's round because agents can chat in any round.

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

**Note:** You must report at most **3 rounds**, separated by commas (e.g., 8, 11, 13). Select rounds representing the most relevant game states where the agent would most benefit from this memory, prioritizing rounds where the memory was NOT retrieved during live play but would have provided significant strategic value. Avoid reporting consecutive or trivial game states; prefer rounds representing key decision points.

Each entry in the Proactive Strategy Report MUST adhere to these structural rules:

- [REMOVE] Strategy: <exact name from database>
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) that contradicts the strategy, if applicable>
  - Reason: <reason>

- [ADD] Strategy: <new strategy name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the strategy should have been applied>
  - Reason: <reason>
  - Objective: <objective description>
  - Policy: <concrete execution steps>

- [MODIFY] Strategy: <exact name from database>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the strategy was applied or considered>
  - Reason: <reason>
  - Field: <Field Name>
    - Old: <current text>
    - New: <replacement text>

- [MERGE] Strategies: <Strategy A Name> + <Strategy B Name>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the unified strategy applies>
  - Reason: <reason>
  - Into Strategy: <new unified strategy name>
  - Objective: <unified objective>
  - Policy: <unified execution steps>

- [KEEP] Strategy: <exact name from database>
  - Type: <either "Chat" or "Action">
  - Round: <comma-separated list of up to 3 round numbers (1-indexed based on Game History markers) where the strategy was successfully deployed>
  - Reason: <reason>

- [GRAVEYARD PROPOSAL]
  - Description: <Concise description of the failed strategy attempt>
  - Policy Flaw: <Extract ONLY the specific part of the strategy's policy that actively harmed the agent or backfired, explaining why>
  (Use this ONLY when identifying an existing Proactive Strategy policy rule that actively harmed the agent and caused a loss, to ensure it is never written again.)


If no notable proactive strategies were observed, write: "No strategies observed."
"""

PROACTIVE_TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Proactive Strategy Database for the agent's own play patterns.
You have just finished {n} game(s). Each Proactive Strategy Report contains feedback tags ([REMOVE], [ADD], [MODIFY], [MERGE]).

--- WHAT IS THE PROACTIVE STRATEGY DATABASE? ---
This database stores the agent's offensive playbook — multi-turn maneuvers, psychological traps, misdirections, and exploits that the agent can proactively initiate to gain a strategic advantage over the opponent.
At inference time, the agent consults this database FIRST at each turn, before considering reactive or verification memory. When a strategy's conditions are met, its 'Policy' is executed to attack or exploit the opponent.
This database is NOT for recording defensive counter-responses or safety checks — those belong in the Reactive Memory Database and Verification Memory Database respectively.

--- CURRENT PROACTIVE STRATEGY DATABASE ---
{current_proactive_ltm}

--- PROACTIVE STRATEGY REPORTS ({n} game(s)) ---
{gradient_reports}

--- PROACTIVE STRATEGY FIELD DEFINITIONS ---
* Strategy Name: A descriptive title for the proactive strategy.
* Type: Must be exactly "Chat" or "Action". This dictates when the strategy fires. If Type is "Chat", the Policy MUST provide instructions on what the agent should say to execute the strategy. If Type is "Action", the Policy MUST provide instructions on what physical move to execute.
* Objective: What this strategy aims to achieve. Maximum 4 sentences.
* Policy: The concrete steps the agent should take to execute the strategy. Maximum 6 sentences.

⚠ CROSS-MODAL RULE: The Type field is just to separate which memory is recalled in each stage (action vs. chat) so the agent can see the policy to fully support what it wants to do at that phase. The whole game state, including both the board and chat history, are context you should make use of when writing the 'Objective' condition. For example, your 'Objective' to generate an Action strategy can be based on some chat behavior from the opponent. Or your 'Objective' for a Chat strategy can be based on the game state being in some form that you can make use of (e.g., the opponent is threatening, so you inject chat to confuse them).

--- APPLICATION RULES ---
Your role is Synthesizer. You MUST execute your task in a strict 2-step process.

STEP 1: GRAVEYARD MANAGEMENT
First, manage any [GRAVEYARD PROPOSAL] entries from the Proactive Strategy Reports.
1. Cluster & Quorum: Group all conceptually identical [GRAVEYARD PROPOSAL]s from the reports. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. A cluster MUST contain at least 2 proposals to meet the quorum. Ignore any proposals that do not meet quorum.
2. Consolidate: Merge the components of each valid cluster into a single Graveyard Proposal.
3. Merge with Existing: Compare the consolidated proposal against the existing "--- GRAVEYARD OF FAILED STRATEGIES ---" (located at the bottom of the current proactive strategy database, if it exists). If an entry with the same underlying description exists, merge them by combining their Policy Flaw lists (ensuring no historical flaws are deleted). Otherwise, prepare to append it as a new entry.

STEP 2: DATABASE SYNTHESIS & CROSS-VERIFICATION
Second, update the main Proactive Strategy Database by applying the standard Proactive Strategy Report tags ([REMOVE], [ADD], [MODIFY], [MERGE], [KEEP]), BUT YOU MUST FIRST FILTER THEM THROUGH THE BATCH QUORUM RULES BELOW.

1. **[REMOVE] (if quorum met)**: Find the named strategy. Delete it entirely.
2. **[ADD] (if quorum met)**: If only one identical ADD is approved, insert it. If multiple similar ADDs are approved, synthesize them into a single unified strategy using the best phrasing from the cluster.
3. **[MODIFY] (if quorum met)**: Find the named strategy. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE] (if quorum met)**: Remove both named strategies. Insert the merged strategy exactly as written.
5. **[KEEP] (if quorum met)**: Record that the named strategy's Policy was vouched for.
6. **ANTI-VAGUENESS RULE**: The Policy MUST describe concrete, executable steps.
7. **NAMING FORMAT RULE**: Strategy names MUST be written in natural language with spaces (e.g., "Chat Noise Suppression"). You are STRICTLY FORBIDDEN from using CamelCase or PascalCase.

--- BATCH QUORUM RULES (apply when {n} > 1) ---
1. **[REMOVE] Threshold**: A strategy MUST receive a [REMOVE] instruction in at least 3 games to be removed. If it appears in <3 games, IGNORE the remove instruction entirely.
2. **[MERGE], [KEEP] Threshold**: These instructions MUST apply to the EXACT same existing strategy name in at least 2 games to be executed. If they appear in only 1 game, IGNORE them entirely.
3. **[MODIFY] Threshold**: For an existing strategy to be modified, conceptually similar [MODIFY] proposals (e.g. adding a similar edge-case exception) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum. If a specific modification is proposed in only a single game, IGNORE that specific modification entirely (even if other, separate modifications to the same strategy met the quorum and are accepted).
4. **[ADD] Threshold**: For a new behavior to be added, conceptually similar [ADD] entries (even if wording or names differ) MUST appear in at least 2 games. IMPORTANT: You must read the content to group them by underlying concept BEFORE counting to check quorum, ignoring differences in their headers or names. If a behavior is observed in only a single game's [ADD], IGNORE it entirely.
5. **NO AUTONOMOUS MERGING**: You are STRICTLY FORBIDDEN from merging strategies on your own. You may only execute a [MERGE] if it was explicitly issued by the gradient reports in at least 2 games. It is better to have multiple specific strategies with good Policy checks than 1 abstract strategy.
6. **STRICT TYPE SEPARATION**: Proposals with different Types ("Chat" vs "Action") are fundamentally distinct. You MUST NEVER group, cluster, or merge a Chat proposal with an Action proposal when counting for quorum or synthesizing.

--- BATCH RECONCILIATION RULES (apply when {n} > 1 and quorum is met) ---
When the same strategy receives conflicting instructions that meet their respective quorum thresholds, resolve as follows:
1. **[KEEP] vs [MERGE]**: [KEEP] takes absolute priority over [MERGE]. If a strategy has proven successful ([KEEP]), DO NOT merge it. Preserve the specific actionable strategy.
2. **[KEEP] vs [REMOVE]**: [KEEP] takes absolute priority over [REMOVE]. A proven successful strategy cannot be removed.
3. **[REMOVE] vs [MODIFY]**: keep the strategy and apply the [MODIFY].
4. **[KEEP] vs [MODIFY] on the Policy field**: If the [MODIFY] adds a conditional exception for a specific edge case, you MUST apply the [MODIFY] to make the rule more robust. Only prefer [KEEP] if the [MODIFY] completely contradicts the original policy without specifying a distinct game-state condition.
5. **[ADD] in multiple games**: synthesize clusters of conceptually similar [ADD]s into one new strategy.
6. **[MODIFY] clusters**: If multiple different valid clusters of modifications (each meeting the 2-game quorum) apply to the same strategy, take the union to cover all valid observations.

--- SYNTHESIS QUALITY RULES ---
- **Role-Agnostic Generalization**: If a strategy is fundamentally applicable regardless of which side, faction, or role you are playing, you MUST write the 'Objective' and 'Policy' fields in a role-agnostic way. Use relative spatial and functional terms (e.g., "your home base", "opponent's starting area", "distance to target", "forward/backward") instead of absolute coordinates, side-specific names, or hardcoded map features. This ensures the strategy remains actionable if you play the opposite side or a different role in future games.
- **Preserve Specificity**: Do not strip concrete tactical details in favor of vague generalizations. It is better to have multiple highly-specific strategies than 1 abstract strategy.
- **Brevity**: The Objective MUST be at most 4 sentences in the final database. The Policy MUST be at most 6 sentences. Distill by removing redundant phrasing — never by dropping distinct tactical conditions or concrete safety checks.
- **NO AUTONOMOUS MERGING**: Do not merge or group strategies unless explicitly commanded by a valid [MERGE] report that meets the quorum.
- **FINAL GUARDRAIL CROSS-VERIFICATION**: After synthesizing the main database, cross-verify it against your updated Graveyard. Ensure that absolutely no policy rules present in the Graveyard have accidentally slipped into the final synthesized database.

You may output as many synthesized memory entries as needed. Each synthesized memory entry MUST use this format:

- Strategy Name: [Short Descriptive Title]
  - Type: [Chat or Action]
  - Objective: [Objective description — max 4 sentences]
  - Policy: [Concrete execution steps — max 6 sentences]

You MUST output the full Graveyard section at the very bottom of your output (carrying over all existing entries and appending any new ones). If no graveyard exists and none was created, you may omit this section:

--- GRAVEYARD OF FAILED STRATEGIES ---
- Description: [Concise description of the failed strategy attempt]
  - Policy Flaw: [The specific part of the policy that actively harmed the agent and why it backfired]

⚠ ACCEPTED PROPOSALS RULE: After writing the full updated signal database, append an
[ACCEPTED] block listing every gradient report proposal that was incorporated.
For EACH newly generated or modified signal in the database, list which original proposals contributed to it using this exact format (you MUST include the Game N tag from the report):
- "[Name of the Strategy as written in the database above]" <- "[Exact Name of Original Proposal 1] [Game 1]", "[Exact Name of Original Proposal 2] [Game 2]"

For example, if the gradient reports provided were:
GAME 1:
- [MODIFY] Strategy: Minimum Competitive Threshold
- [MERGE] Strategies: Alpha + Beta
  - Into Strategy: Unified Defense
GAME 2:
- [MODIFY] Strategy: Minimum Competitive Threshold
- [MERGE] Strategies: Alpha + Beta
  - Into Strategy: Unified Defense

Then your accepted block must look exactly like this:
[ACCEPTED]
- "Minimum Competitive Threshold" <- "[MODIFY] Strategy: Minimum Competitive Threshold [Game 1]", "[MODIFY] Strategy: Minimum Competitive Threshold [Game 2]"
- "Unified Defense" <- "[MERGE] Strategies: Alpha + Beta [Game 1]", "[MERGE] Strategies: Alpha + Beta [Game 2]"

Write ONLY the full updated proactive strategy memory, graveyard, and the [ACCEPTED] block. Do not include any pleasantries or conversational filler.
If no proactive strategy memory exists yet and the gradient report contains ADD strategies, write a fresh database from those strategies.
If the final synthesized database is completely empty (i.e., no strategies are currently stored), you MUST output exactly:
(No strategies currently stored)
Do not output any explanation, reasoning, or other text when the database is empty.
"""


