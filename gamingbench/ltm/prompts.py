LTM_INJECTION_PROMPT = """\
=== YOUR OPPONENT REPUTATION DATABASE ===
From your experience in previous games against {opponent_id}, you have accumulated the following behavioral knowledge about this specific opponent.

--- HOW TO READ THESE ENTRIES ---
Each entry describes a behavioral pattern that {opponent_id} has repeatedly exhibited across past games. The fields mean:
- Signal: A short name for the recurring behavioral pattern observed in {opponent_id}.
- When: The trigger condition — the situation or game state in which {opponent_id} exhibits this behavior.
  ⚠ NOTE: Use this as a prior and verify against live observations this game.
- What: A generalized description of {opponent_id}'s behavior. Use this to recognize when the pattern is occurring.
- Policy: The mandatory action to execute when you detect this pattern. This is not a suggestion — once the signal fires, the Policy MUST be followed exactly.

--- OPPONENT REPUTATION DATABASE ---
{ltm_text}

--- HOW TO USE THIS DATABASE ---
1. Identify which signals are firing: Match {opponent_id}'s live actions against the When and What fields to detect active signals.
2. Treat When/What as probabilistic priors, not facts: A signal's trigger condition adjusts your expectation of opponent behavior — it is not proof. If live evidence strongly contradicts the When/What prediction, you may down-weight the signal.
   ⚠ CRITICAL EXCEPTION: Rule 2 applies ONLY to the When/What prediction fields. It does NOT apply to the Policy field. Once a signal's trigger is confirmed, the Policy is a mandatory executable action, not a suggestion. You may NOT override the Policy because you expect a better outcome from a different action.
3. Execute the Policy: When a signal fires, follow its Policy field exactly. The policy encodes decisions refined across many past games and supersedes in-game reasoning about expected outcomes.
=== END OPPONENT REPUTATION DATABASE ===
"""

WINDOW_SUMMARIZE_PROMPT = """\
The last {K} moves just completed. Based on everything you have observed and decided this window, output exactly two lines:

⚠ IMPORTANT: DO NOT select or output a move in your response. Your ONLY task right now is to generate this summary.

Game/Opponent summary: [A few sentences — key observations about the opponent's moves and behavior that you need for post-game evaluation. If chat is active, compare what the opponent communicated against their actual moves to surface any relevant patterns.]

Reasoning memory: [A few sentences — the core reasoning behind your own key moves this window. If any signal from the OPPONENT REPUTATION DATABASE influenced a decision, explicitly name the exact Signal and describe how you applied its Policy and what the outcome was on the board.]

⚠ NOTE: If no OPPONENT REPUTATION DATABASE is present, focus entirely on describing your own reasoning and observations this window.
"""

GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game against a specific opponent.
Your goal is to compare the agent's in-game observations against the GROUND TRUTH game history and produce a structured gradient report.

--- ✅ MATCH GROUND TRUTH (Full History) ---
These are the confirmed, objective move-by-move outcomes. Use these as the authoritative ground truth.
{game_history}

--- AGENT'S IN-GAME OBSERVATIONS (Window Summaries) ---
These are the agent's own observations and strategic thoughts recorded during the game.
{window_summaries}

--- CURRENT LONG-TERM MEMORY (Prior) ---
This is what the agent believed about this opponent BEFORE the game started.
{current_ltm}
("(No memory yet)" means this is the first game — treat all signals as ABSENT.)

You are building a GRADIENT REPORT for the agent's Opponent Reputation Database.
The goal of this report is to improve the agent's knowledge of this opponent so that in future games the agent can maximize its win rate. Each proposed update should bring the database closer to an accurate, complete, and strategically actionable understanding of the opponent — closing knowledge gaps that, if resolved, would unlock better strategies. A high-quality report captures BOTH types of signals: (1) Harm signals — opponent behaviors that damaged, deceived, or outmaneuvered the agent; and (2) Exploitable Weakness signals — patterns in the opponent's play that the agent successfully exploited, or that represented missed opportunities the agent should target in future games.

⚠ CHAT ANALYSIS: If a FULL CHAT TRANSCRIPT is provided in the game history, it is a critical source of observation. Explicitly compare what the opponent communicated versus their physical moves to determine if they tend to bluff, negotiate honestly, or manipulate.

--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* When: The trigger condition — the full observable context and prior state that preceded or coincided with this behavior. Describe everything that could plausibly have driven the opponent's decision: the situational history, established patterns, and any relevant observable signals present at the time. Do not infer triggers that were not directly observed.
* What: The factual observation of what the opponent did. Write only what was directly observed. Never use conditional language (e.g., "as long as", "whenever", "unless") — those imply rules that may not have been tested. If a condition was not tested, state that explicitly.
* Policy: The executable action plan that aims to maximize the agent's win rate.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.

Analyze the game and propose updates using the following 4 tags:
- [REMOVE]: Identify a signal whose core observed behavior (When/What) is directly contradicted by the ground truth, or whose entire signal — even after potential modification — is net harmful to retain. Do NOT use [REMOVE] if only the Policy is wrong; use [MODIFY] to fix the Policy instead. Do NOT use [REMOVE] simply because a signal's trigger was not encountered this game — absence of evidence is not contradiction.
- [ADD]: Define a completely new observed behavior not yet covered by any signal in the current database. If an existing signal partially covers the behavior but one or more fields are incorrect, use [MODIFY] instead of adding a duplicate.
- [MODIFY]: Identify an existing signal worth keeping, but whose one or more fields are inaccurate, misleading, too general, or too specific based on the current game evidence. Prefer [MODIFY] over [ADD] when the behavior is already partially captured by an existing signal. Only list the fields that are changing; omit all unchanged fields.
- [MERGE]: Identify two existing signals that share the same trigger condition or consistently co-occur, and would produce a clearer and more actionable unified policy as a single signal. Do NOT merge signals that share a surface theme but have different triggers or require different responses — merging those would weaken both policies.

You may include as many update entries as necessary. A single gradient report can contain multiple [REMOVE]s, multiple [ADD]s, multiple [MODIFY]s, etc., depending on what the game data supports.

⚠ OPPONENT-BEHAVIOR-ONLY RULE: Every signal you report — whether REMOVE, ADD, MODIFY, or MERGE — MUST describe a behavioral pattern of the OPPONENT, not the agent's own strategy. Concretely: the When and What fields must be grounded in observable actions taken by the opponent during this game. 

⚠ VERIFICATION RULE: Do not assert untested opponent behavior. You must only describe behaviors and responses that were explicitly triggered and observed in the current game. If a specific action was never taken by the agent, you cannot make claims about how the opponent would have reacted to it.

⚠ PRESERVATION RULE: When proposing a [MODIFY] on ANY field, do NOT replace the Old value with a narrower description covering only what was triggered this game. The Current LTM was built by accumulating evidence across many past games, each testing different conditions. If an Old field describes behaviors or conditions confirmed in past games but not re-triggered this game, those observations remain valid. A [MODIFY] on any field is only warranted if this game: (a) adds new observations that should be incorporated into the existing description, or (b) directly contradicts a specific claim in the Old value. 

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

Before writing your entries, answer these two questions using the ground truth above:
- Were there moments where the opponent's behavior directly hurt the agent (deception, captures, manipulation)? → Capture these as Harm signals.
- Were there moments where the opponent's strategy had a clear gap or predictable pattern the agent exploited or could have exploited? → Capture these as Exploitable Weakness signals.
Your gradient report is incomplete if it only contains Harm signals and no Exploitable Weakness signals (or vice versa), unless the game genuinely provided no evidence for one type.

Each entry in the gradient report MUST adhere to these structural rules:

- [REMOVE] Signal: <exact name from database>
  - Reason: <one sentence citing the specific ground truth observation that contradicts this signal>

- [ADD] Signal: <new signal name>
  - When: <specific trigger condition observation>
  - What: <specific behavior observation>
  - Policy: <concrete executable action>

- [MODIFY] Signal: <exact name from database>
  - Field: <Name of Field to Change, e.g., When, What, or Policy>
    - Old: <current text>
    - New: <replacement text>
  (List only the fields that are changing. Omit unchanged fields.)

- [MERGE] Signals: <Signal A Name> + <Signal B Name>
  - Into Signal: <new unified signal name>
  - When: <unified trigger condition>
  - What: <unified behavior description>
  - Policy: <concrete executable unified policy>

⚠ ANTI-VAGUENESS RULE: The policy MUST name a concrete, executable action. 

--- EXAMPLES OF VALID GRADIENT ENTRIES ---

- [ADD] Signal: Early Chat Bluffing
  - When: The opponent initiates chat in the first 3 moves.
  - What: They claim to be a beginner or pretend to make a mistake.
  - Policy: Ignore their chat claims completely and assume they are an experienced player attempting to bait an overextension.

- [MODIFY] Signal: Central Pawn Push
  - Field: Policy
    - Old: Block the pawn immediately.
    - New: Ignore the central pawn and advance your edge pawns to create a dual threat.

- [REMOVE] Signal: Avoids A-Column
  - Reason: The opponent advanced their A-column pawn in move 12, contradicting the prior belief that they ignore the left edge.

If no notable signals were observed, write: "No signals observed."
Do NOT rewrite the current Long-Term Memory. Only produce the gradient report.
"""

TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Opponent Reputation Database for playing against a specific opponent.
You have just finished a game. Each gradient report contains feedback tags for behavioral signals:
- [REMOVE]: Signals that have been invalidated by the ground truth.
- [ADD]: New signals to add to the database.
- [MODIFY]: Existing signals with fields that need specific updating.
- [MERGE]: Signals that should be combined into a single new signal.

--- CURRENT OPPONENT REPUTATION DATABASE ---
{current_ltm}

--- GRADIENT REPORT FROM LATEST GAME ---
{gradient_report}

--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* When: The specific trigger condition or game state that causes the behavior.
* What: A description of the opponent's behavior.
* Policy: The executable action plan that aims to maximize the agent's win rate.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.

Your role is Synthesizer. Update the Opponent Reputation Database by applying the gradient report using the following rules:

1. **[REMOVE]**: Find the named signal in the current database. Delete it entirely.
2. **[ADD]**: Insert the new signal exactly as written in the gradient report. No changes.
3. **[MODIFY]**: Find the named signal. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE]**: Remove both named signals. Insert the merged signal exactly as written.
5. **ANTI-VAGUENESS RULE**: The policy MUST name a concrete, executable action. Reject any policy that could apply generically to any opponent (e.g., "be cautious", "pay attention to their behavior").


Each synthesized memory entry MUST use this format:

- Signal: [Short Name of Pattern]
  - When: [Specific trigger condition observation]
  - What: [Specific behavior observation]
  - Policy: [Concrete executable action]

Write ONLY the updated memory. Do not include any pleasantries or conversational filler.
If no memory exists yet and the gradient report contains ADD signals, write a fresh memory from those signals.
"""
