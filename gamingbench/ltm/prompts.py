LTM_INJECTION_PROMPT = """\
=== YOUR OPPONENT REPUTATION DATABASE ===
From your experience in previous games against {opponent_id}, you have accumulated the following behavioral knowledge about this specific opponent.

--- HOW TO READ THESE ENTRIES ---
Each entry describes a behavioral pattern that {opponent_id} has repeatedly exhibited across past games. The fields mean:
- Signal: A short name for the recurring behavioral pattern observed in {opponent_id}.
- When: The trigger condition — the situation or game state that activates this signal. This is the SOLE criterion for determining whether a signal fires.
- What: A description of what {opponent_id} is likely attempting once this signal fires. Use this to understand their strategy and anticipate their intent — NOT to decide whether the signal fires.
- Policy: The mandatory action to execute when you detect this pattern. This is not a suggestion — once the signal fires, the Policy MUST be followed exactly.

--- OPPONENT REPUTATION DATABASE ---
{ltm_text}

--- HOW TO USE THIS DATABASE ---
1. Identify which signals are firing: Match the current game state and context strictly against the 'When' field. A signal fires as soon as its 'When' condition is met — do NOT wait for the opponent to execute the predicted 'What' behavior before activating the policy. By then it may be too late.
2. Treat 'When' as a probabilistic prior, not a fact: A signal's trigger condition adjusts your expectation of opponent behavior — it is not proof. If live evidence strongly contradicts the 'When' condition, you may down-weight the signal.
   ⚠ CRITICAL EXCEPTION: Rule 2 applies ONLY to the 'When' trigger field. It does NOT apply to the Policy field. Once a signal's trigger is confirmed, the Policy is a mandatory executable action, not a suggestion. You may NOT override the Policy simply because you expect a better outcome from a different action.
   ⚠ POLICY HARM EXCEPTION: The one permitted override is when executing the Policy-prescribed action would be directly harmful to your position in the current game state — meaning the action itself actively worsens your standing (e.g., it allows the opponent an immediate decisive advantage, or it forces you into a self-defeating move). If the prescribed action is neutral or beneficial, you MUST follow the Policy. This exception is NOT a license to ignore the Policy on general strategic grounds — it applies only when the prescribed action is concretely harmful right now, not merely suboptimal by your in-game reasoning.
3. Use 'What' for strategic understanding: Once a signal fires, read the 'What' field to understand what the opponent is trying to accomplish. Use this to anticipate their next move while executing your Policy.
4. Execute the Policy: When a signal fires, follow its Policy field exactly. The policy encodes decisions refined across many past games and supersedes in-game reasoning about expected outcomes.
=== END OPPONENT REPUTATION DATABASE ===
"""

WINDOW_SUMMARIZE_PROMPT = """\
⚠ IMPORTANT: DO NOT select or output a move in your response. Your ONLY task right now is to generate a summary.

The last {K} moves just completed. Based on everything you have observed and decided this window, output EXACTLY the following two sections:

Game/Opponent summary: [A few sentences — key observations about the opponent's moves and behavior that you need for post-game evaluation. If chat is active, compare what the opponent communicated against their actual moves to surface any relevant patterns.]

Reasoning memory: [A few sentences — the core reasoning behind your own key moves this window.
  (1) Opponent signals: You MUST enumerate every Signal from the OPPONENT REPUTATION DATABASE that you used and state (a) which specific move you played in response to its Policy and what the immediate board outcome was, (b) what do you think about this signal — did it effectively give you an upperhand, was it neutral, or did it harm you. Also include any opponent signal whose trigger you observed but whose Policy you chose not to follow, explaining why.
  (2) Self signals: You MUST enumerate every Signal from the SELF-REPUTATION DATABASE that fired this window and state (a) whether you successfully followed the corrective Policy (for FLAW signals) or reinforced the effective tactic (for STRENGTH signals), and (b) what the board outcome was. Also include any self signal whose trigger you observed but whose Policy you did not follow, explaining why.]

⚠ NOTE: If no OPPONENT REPUTATION DATABASE or SELF-REPUTATION DATABASE is present, focus entirely on describing your own reasoning and observations this window.
"""

GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game against a specific opponent.
Your goal is to compare the agent's in-game observations against the GROUND TRUTH game history and produce a structured gradient report.

--- ✅ MATCH GROUND TRUTH (Full History) ---
These are the confirmed, objective move-by-move outcomes. Use these as the authoritative ground truth.

Reading the history:
{game_history_legend}
- [Chat]: Chat message sent that turn (if chat is enabled).
- [Move]: The physical move executed after the position above.

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


--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* When: The trigger condition — the full observable context and prior state that preceded or coincided with this behavior. Describe everything that could plausibly have driven the opponent's decision: the situational history, established patterns, and any relevant observable signals present at the time. Do not infer triggers that were not directly observed. Maximum 4 sentences.
* What: The factual observation of what the opponent did. Write only what was directly observed. Never use conditional language (e.g., "as long as", "whenever", "unless") — those imply rules that may not have been tested. If a condition was not tested, state that explicitly. Maximum 4 sentences.
* Policy: The executable action plan that aims to maximize the agent's win rate. Maximum 4 sentences.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.

Analyze the game and propose updates using the following 5 tags:
- [REMOVE]: Identify a signal whose core observed behavior (When/What) is directly contradicted by the ground truth, or whose entire signal — even after potential modification — is net harmful to retain. Do NOT use [REMOVE] if only the Policy is wrong; use [MODIFY] to fix the Policy instead. Do NOT use [REMOVE] simply because a signal's trigger was not encountered this game — absence of evidence is not contradiction.
- [ADD]: Define a completely new observed behavior not yet covered by any signal in the current database. If an existing signal partially covers the behavior but one or more fields are incorrect, use [MODIFY] instead of adding a duplicate.
- [MODIFY]: Identify an existing signal worth keeping, but whose one or more fields are inaccurate, misleading, too general, or too specific based on the current game evidence. Prefer [MODIFY] over [ADD] when the behavior is already partially captured by an existing signal. Only list the fields that are changing; omit all unchanged fields.
- [MERGE]: Identify two or more existing signals that are variations of the same underlying behavior. Use [MERGE] liberally: two signals should be merged if a single, unified policy would serve the agent equally well in both triggering situations. Prefer [MERGE] over a combination of [MODIFY]+[ADD] whenever the new game evidence is already partially captured by two separate existing signals that happen to co-occur or have similar triggers. Do NOT limit merging to signals with identical trigger wording — trigger conditions are considered equivalent if they describe the same underlying game situation from different angles. Do NOT merge signals that require clearly distinct responses where unifying the policy would weaken one of them.
- [KEEP]: Emit this tag when a signal's Policy was explicitly executed in this game AND doing so was causally beneficial to the agent winning. [KEEP] is a positive vouching action — emit it only when you are confident the Policy deserves credit for the outcome. Do NOT emit [KEEP] merely because the agent won; the signal's Policy must have been directly followed and must have contributed to the win. Do NOT emit [KEEP] if the game was a draw or a loss. Unmentioned signals carry no implication — [KEEP] is not a default; it is a deliberate endorsement.
  ⚠ [KEEP] protects only the Policy field of the named signal. You may still emit a concurrent [MODIFY] on the same signal's When or What fields if those fields need updating — [KEEP] will not block those changes. Only the Policy field is shielded.

You may include as many update entries as necessary. A single gradient report can contain multiple [REMOVE]s, multiple [ADD]s, multiple [MODIFY]s, multiple [KEEP]s, etc., depending on what the game data supports.

⚠ OPPONENT-BEHAVIOR-ONLY RULE: Every signal you report — whether REMOVE, ADD, MODIFY, MERGE, or KEEP — MUST describe a behavioral pattern of the OPPONENT, not the agent's own strategy. Concretely: the When and What fields must be grounded in observable actions taken by the opponent during this game. 

⚠ VERIFICATION RULE: Do not assert untested opponent behavior. You must only describe behaviors and responses that were explicitly triggered and observed in the current game. If a specific action was never taken by the agent, you cannot make claims about how the opponent would have reacted to it.

⚠ INFORMATION FIDELITY RULE: When modifying any field, your goal is to produce the most accurate, general description that still captures every confirmed observation from past games. Before writing a [MODIFY] on 'When' or 'What', apply this test: "Does the new text still fire (or describe) the same situations the old text covered, and is the Policy still correct in all those situations?" If yes, prefer the more concise or general form. If no, keep the more specific wording.
  - Do NOT replace the 'When' trigger with a narrower one that drops previously confirmed trigger conditions — that is always a loss of information.
  - DO replace an overfitted trigger with a broader abstraction if it cleanly subsumes all previously confirmed cases without losing any — that is an improvement, not a loss.
  - For 'What', distill confirmed observations into the most concise, general behavioral description. You MAY drop specific observed instances if a generalized abstraction fully covers them. A 'What' that grows longer with each game is a failure mode; aim to converge toward a shorter, more accurate description over time.

⚠ ANTI-DUPLICATION RULE: You are STRICTLY FORBIDDEN from using [ADD] if the core concept is already represented in the database. You MUST use [MODIFY] to elevate the existing signal to a higher level of abstraction that successfully covers the new edge case. [ADD] is reserved exclusively for fundamentally new behaviors that cannot be logically grouped with any existing signal.

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

⚠ PRE-ANALYSIS (complete both steps in your internal reasoning before writing any entries):
1. AGENT SIGNAL AUDIT: Go through each window summary in the AGENT'S IN-GAME OBSERVATIONS above and extract: (a) which LTM signals the agent explicitly activated this game, (b) which Policy actions were actually executed as a result, and (c) what the board outcome was. These are the only link you have to how the reputation memory was used in-game, so analyze them carefully. If a signal's Policy was followed but the outcome was poor or neutral, prioritize a [MODIFY] on that signal's Policy so the agent does not repeat the same mistake. If the agent won and a signal's Policy was directly executed and contributed to the win, consider emitting [KEEP] for that signal.
2. CHAT ANALYSIS (only if a chat transcript is present in the game history): Explicitly compare what the opponent communicated versus their physical moves to determine if they tend to bluff, negotiate honestly, or manipulate. If no chat is present, skip this step.


Each entry in the gradient report MUST adhere to these structural rules:

- [REMOVE] Signal: <exact name from database>
  - Reason: <one or two sentences citing the specific ground truth observation that contradicts this signal>

- [ADD] Signal: <new signal name>
  - Reason: <one or two sentences explaining why this new signal is warranted and not covered by any existing entry>
  - When: <specific trigger condition observation — max 4 sentences>
  - What: <specific behavior observation — max 4 sentences>
  - Policy: <concrete executable action — max 4 sentences>

- [MODIFY] Signal: <exact name from database>
  - Reason: <one or two sentences explaining what evidence from this game justifies the change>
  - Field: <Name of Field to Change, e.g., When, What, or Policy>
    - Old: <current text>
    - New: <replacement text>
  (List only the fields that are changing. Omit unchanged fields.)

- [MERGE] Signals: <Signal A Name> + <Signal B Name>
  - Reason: <one or two sentences explaining why a single unified policy serves both triggering situations equally well>
  - Into Signal: <new unified signal name>
  - When: <unified trigger condition — max 4 sentences>
  - What: <unified behavior description — max 4 sentences>
  - Policy: <concrete executable unified policy — max 4 sentences>

- [KEEP] Signal: <exact name from database>
  - Reason: <one or two sentences explaining how this signal's Policy was executed this game and why it was causally beneficial to the agent's win>

⚠ ANTI-VAGUENESS RULE: The policy MUST name a concrete, executable action. 
⚠ BREVITY RULE: Each of When, What, and Policy MUST be at most 4 sentences.

If no notable signals were observed, write: "No signals observed."
Do NOT rewrite the current Long-Term Memory. Only produce the gradient report.

After your gradient entries, output a second mandatory section with this exact heading:

### Correctness Scores

For each signal in the CURRENT LONG-TERM MEMORY that is being RETAINED (i.e., not [REMOVE]d and not a [MERGE] source signal), output exactly one line:
  - Signal: <exact signal name from database> → <LABEL>

Choose exactly one label per signal based strictly on the ground truth game history:
  - CONFIRMED              : The signal's trigger (When) fired this game AND the opponent's behavior (What) matched exactly.
  - MOSTLY_CONFIRMED       : The trigger fired AND behavior mostly matched, with only minor deviations.
  - ABSENT                 : The trigger condition did not occur this game at all. Use this when the signal had no opportunity to fire.
  - PARTIALLY_CONTRADICTED : The trigger fired BUT the opponent's behavior only partially matched the What description.
  - CONTRADICTED           : The trigger fired AND the opponent's behavior clearly contradicted the What description.

⚠ Do NOT include [REMOVE] signals, [MERGE] source signals, or newly [ADD]ed signals in this section.
⚠ [ADD] and [MERGE] result signals are scored automatically by the system — do not list them here.
⚠ If there are no retained existing signals to score, write: "No signals to score."

Example:
### Correctness Scores
- Signal: Deceptive Symmetrical Chat Agreements → CONFIRMED
- Signal: Masking Home-Row Pushes → ABSENT
"""

TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Opponent Reputation Database for playing against a specific opponent.
You have just finished {n} game(s). Each gradient report contains feedback tags for behavioral signals:
- [REMOVE]: Signals that have been invalidated by the ground truth.
- [ADD]: New signals to add to the database.
- [MODIFY]: Existing signals with fields that need specific updating.
- [MERGE]: Signals that should be combined into a single new signal.

--- CURRENT OPPONENT REPUTATION DATABASE ---
{current_ltm}

--- GRADIENT REPORTS ({n} game(s)) ---
{gradient_reports}

--- LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern.
* When: The specific trigger condition or game state that causes the behavior.
* What: A description of the opponent's behavior.
* Policy: The executable action plan that aims to maximize the agent's win rate.
  - If all relevant opponent behavior has been observed: prescribe the optimal exploitation action directly.
  - If the What field notes untested conditions that, if known, would enable a better strategy: prescribe (1) how and when to probe for that missing information, and (2) what action to take contingent on the probe result.
  - The Policy MUST NOT assume untested opponent behavior when prescribing actions.

--- APPLICATION RULES ---
Your role is Synthesizer. Update the Opponent Reputation Database by applying the gradient report(s).
Note: each gradient entry includes a Reason field for your context. Use the Reason to better understand the intent and evidence behind an instruction, but do not copy Reason fields into the final database output.

1. **[REMOVE]**: Find the named signal in the current database. Delete it entirely.
2. **[ADD]**: Insert the new signal exactly as written in the gradient report. No changes.
3. **[MODIFY]**: Find the named signal. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE]**: Remove both named signals. Insert the merged signal exactly as written.
5. **[KEEP]**: Record that the named signal's Policy was vouched for as causally beneficial in a winning game. The Policy field of this signal is protected — see reconciliation rule 8 below for how to apply this when conflicts arise.
6. **ANTI-VAGUENESS RULE**: The policy MUST name a concrete, executable action. Reject any policy that could apply generically to any opponent (e.g., "be cautious", "pay attention to their behavior").

--- BATCH RECONCILIATION RULES (apply when {n} > 1) ---
When the same signal receives conflicting instructions across the {n} gradient reports, resolve as follows:

1. **[REMOVE] vs [MODIFY]** on the same signal → keep the signal and apply the [MODIFY].
2. **[ADD] in only 1 game** → include it as a new signal; if an existing signal in the database describes nearly identical behavior, merge them into one generalized entry — otherwise keep it separate.
3. **[ADD] in multiple games** → for each cluster of [ADD] entries that describe the same behavioral pattern, synthesize that cluster into one generalized entry; [ADD] entries describing distinct behaviors each become their own separate signal.
4. **[MODIFY] on the same field, consistent direction across games** → merge into a single coherent New value.
5. **[MODIFY] on the same field, contradicting directions** → take the union; abstract to the more general condition that covers both observations.
6. **[MERGE] where a named signal does not exist** → convert to [MODIFY] on the surviving signal.
7. **[MERGE] in some games, [MODIFY] in others on the same signals** → execute the [MERGE] and fold in the modifications.
8. **[KEEP] vs [MODIFY] on the Policy field of the same signal** → apply the [MODIFY] only if it makes the Policy more specific or correct without changing the core prescribed action. If the [MODIFY] would change the core executable action itself (i.e., prescribe a fundamentally different response), prefer [KEEP] and discard that [MODIFY] on the Policy field. [KEEP] does NOT block [MODIFY] on the When or What fields — those may be updated freely.

--- SYNTHESIS QUALITY RULES ---
- **Abstract and Generalize**: Do not include game-specific details (specific round numbers, one-off board states). Produce generalized behavioral descriptions that apply across games.
- **Brevity**: Each of When, What, and Policy MUST be at most 4 sentences in the final database. 
- **Merge-First Discipline**: Before finalizing the output, scan the *entire* resulting database for signals that describe variations of the same underlying behavior. Two signals should be merged into one if a single unified policy would serve the agent equally well in both triggering situations. Do not require identical trigger wording — triggers are equivalent if they describe the same underlying game situation from different angles. Emit a [MERGE] for any such pair you identify, even if the gradient reports did not propose one.
- **ANTI-DUPLICATION RULE**: You are STRICTLY FORBIDDEN from finalizing an [ADD] if the core concept is already represented in the database. You MUST instead use [MODIFY] or [MERGE] to elevate the existing signal to a higher level of abstraction that covers the new edge case.
- **Distinct signals only**: Only retain a signal as a separate entry if its trigger condition and required response are clearly distinct from every other signal in the database such that merging would weaken the policy.


Each synthesized memory entry MUST use this format:

- Signal: [Short Name of Pattern]
  - When: [Specific trigger condition observation — max 4 sentences]
  - What: [Specific behavior observation — max 4 sentences]
  - Policy: [Concrete executable action — max 4 sentences]

Write ONLY the updated memory. Do not include any pleasantries or conversational filler.
If no memory exists yet and the gradient report contains ADD signals, write a fresh memory from those signals.
"""


SELF_LTM_INJECTION_PROMPT = """\
=== YOUR SELF-REPUTATION DATABASE ===
From your experience in previous games, you have accumulated the following knowledge about your own recurring play patterns.

--- HOW TO READ THESE ENTRIES ---
Each entry describes a behavioral pattern you have repeatedly exhibited across past games. The fields mean:
- Signal: A short name for the recurring pattern in your own play.
- Type: FLAW (a recurring mistake to correct) or STRENGTH (an effective tactic to reinforce).
- When: The board state or situation that activates this signal. This is the SOLE criterion for determining whether a signal fires.
- What: For FLAW: what you tend to do incorrectly when this situation arises. Use this for self-recognition — to notice you are about to repeat the bad habit.
        For STRENGTH: what you do effectively in this situation. Use this to intentionally reproduce the tactic.
  In both cases, use 'What' for self-awareness ONLY — NOT to decide whether the signal fires.
- Policy: For FLAW: the corrective action to execute instead of the What behavior.
          For STRENGTH: confirmation to continue and reinforce the What behavior.
  This is mandatory once the signal fires.

--- SELF-REPUTATION DATABASE ---
{self_ltm_text}

--- HOW TO USE THIS DATABASE ---
1. Identify which signals are firing: Match the current board state and game context strictly against the 'When' field. A signal fires as soon as its 'When' condition is met — do NOT wait to observe the 'What' behavior before activating the Policy.
2. Treat 'When' as a probabilistic prior, not a fact: If live evidence strongly contradicts the 'When' condition, you may down-weight the signal.
   ⚠ CRITICAL EXCEPTION: Rule 2 applies ONLY to the 'When' trigger field. It does NOT apply to the Policy field. Once a signal's trigger is confirmed, the Policy is a mandatory executable action, not a suggestion. You may NOT override it simply because you expect a better outcome from a different action.
   ⚠ POLICY HARM EXCEPTION: The one permitted override is when executing the Policy-prescribed action would be directly harmful to your position in the current game state — meaning the action itself actively worsens your standing (e.g., it allows the opponent an immediate decisive advantage, or it forces you into a self-defeating move). If the prescribed action is neutral or beneficial, you MUST follow the Policy. This exception is NOT a license to ignore the Policy on general strategic grounds — it applies only when the prescribed action is concretely harmful right now, not merely suboptimal by your in-game reasoning.
3. Use 'What' for self-awareness: Once a signal fires, read the 'What' field to understand your own pattern — either to consciously avoid it (FLAW) or to intentionally reproduce it (STRENGTH).
4. Execute the Policy exactly as written. For FLAW signals this corrects your tendency; for STRENGTH signals this reinforces your advantage.
=== END SELF-REPUTATION DATABASE ==="""


SELF_GRADIENT_ENGINE_PROMPT = """\
You are an advanced strategy analyzer evaluating an agent's performance in a completed game.
Your goal is to compare the agent's in-game decisions against the GROUND TRUTH game history and produce a structured self-gradient report for the agent's own Self-Reputation Database.

--- ✅ MATCH GROUND TRUTH (Full History) ---
These are the confirmed, objective move-by-move outcomes. Use these as the authoritative ground truth.

Reading the history:
{game_history_legend}
- [Chat]: Chat message sent that turn (if chat is enabled).
- [Move]: The physical move executed after the position above.

{game_history}

--- AGENT'S IN-GAME OBSERVATIONS (Window Summaries) ---
These are the agent's own observations and strategic thoughts recorded during the game.
{window_summaries}

--- CURRENT SELF-REPUTATION DATABASE (Prior) ---
This is what the agent knew about its own patterns BEFORE this game started.
{current_self_ltm}
("(No self-memory yet)" means this is the first game — treat all signals as ABSENT.)

You are building a SELF-GRADIENT REPORT for the agent's Self-Reputation Database.
The goal is to improve the agent's self-awareness so it can avoid recurring mistakes (FLAW signals) and reliably reproduce effective tactics (STRENGTH signals) in future games.

A high-quality report captures BOTH types of signals:
(1) FLAW signals — patterns where the agent's decisions were poor, cost material, or missed winning opportunities.
(2) STRENGTH signals — patterns in the agent's play that were effective and should be reinforced, particularly non-obvious tactics discovered through gameplay experience.

--- SELF-LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern in the agent's own play.
* Type: FLAW or STRENGTH.
* When: The trigger condition — the board state or game context that activates this signal. Describe only directly observable game state, not inferences about the opponent. Maximum 4 sentences.
* What: For FLAW: what the agent typically does incorrectly in this situation. For STRENGTH: what the agent does effectively. Write only what was directly observed. Maximum 4 sentences.
* Policy: For FLAW: the corrective action to execute instead. For STRENGTH: confirmation to reinforce the tactic. Maximum 4 sentences.
  - The Policy MUST be a concrete, executable action.
  - The Policy MUST NOT describe opponent behavior — it must prescribe only what the agent itself should do.

Analyze the game and propose updates using the following 5 tags:
- [REMOVE]: Identify a signal whose core observed behavior (When/What) is directly contradicted by the ground truth, or whose signal is net harmful to retain. Do NOT use [REMOVE] if only the Policy is wrong; use [MODIFY] instead. Do NOT use [REMOVE] simply because a signal's trigger was not encountered this game.
- [ADD]: Define a completely new self-pattern not yet covered by any signal. If an existing signal partially covers it, use [MODIFY] instead.
- [MODIFY]: Identify an existing signal worth keeping but with inaccurate, misleading, too general, or too specific fields. Only list fields that are changing.
- [MERGE]: Identify two or more existing signals that are variations of the same underlying behavior. Use [MERGE] liberally: two signals should be merged if a single, unified policy would serve the agent equally well in both triggering situations. Prefer [MERGE] over a combination of [MODIFY]+[ADD] whenever the new game evidence is already partially captured by two separate existing signals that happen to co-occur or have similar triggers. Do NOT limit merging to signals with identical trigger wording — trigger conditions are considered equivalent if they describe the same underlying game situation from different angles. Do NOT merge signals that require clearly distinct responses where unifying the policy would weaken one of them.
- [KEEP]: Emit this tag when a self-signal's Policy was explicitly executed in this game AND doing so was causally beneficial to the agent winning. [KEEP] is a positive vouching action — emit it only when you are confident the Policy deserves credit for the outcome. Do NOT emit [KEEP] merely because the agent won; the signal's Policy must have been directly followed and must have contributed to the win. Do NOT emit [KEEP] if the game was a draw or a loss. Unmentioned signals carry no implication — [KEEP] is not a default; it is a deliberate endorsement.
  ⚠ [KEEP] protects only the Policy field of the named signal. You may still emit a concurrent [MODIFY] on the same signal's When or What fields if those fields need updating — [KEEP] will not block those changes. Only the Policy field is shielded.

You may include as many update entries as necessary.

⚠ AGENT-BEHAVIOR-ONLY RULE: Every signal you report — whether REMOVE, ADD, MODIFY, MERGE, or KEEP — MUST describe a pattern in the AGENT'S OWN play, not the opponent's behavior. The When and What fields must be grounded in the agent's own actions and board positions.

⚠ VERIFICATION RULE: Do not assert unobserved agent behavior. Only describe patterns that were explicitly triggered and observed in the current game.

⚠ INFORMATION FIDELITY RULE: When modifying any field, your goal is to produce the most accurate, general description that still captures every confirmed observation from past games. Before writing a [MODIFY] on 'When' or 'What', apply this test: "Does the new text still fire (or describe) the same situations the old text covered, and is the Policy still correct in all those situations?" If yes, prefer the more concise or general form. If no, keep the more specific wording.
  - Do NOT replace the 'When' trigger with a narrower one that drops previously confirmed trigger conditions — that is always a loss of information.
  - DO replace an overfitted trigger with a broader abstraction if it cleanly subsumes all previously confirmed cases without losing any — that is an improvement, not a loss.
  - For 'What', distill confirmed observations into the most concise, general description. You MAY drop specific observed instances if a generalized abstraction fully covers them. A 'What' that grows longer with each game is a failure mode; aim to converge toward a shorter, more accurate description over time.

⚠ ANTI-DUPLICATION RULE: You are STRICTLY FORBIDDEN from using [ADD] if the core concept is already represented in the database. You MUST use [MODIFY] to elevate the existing signal to a higher level of abstraction that successfully covers the new edge case. [ADD] is reserved exclusively for fundamentally new behaviors that cannot be logically grouped with any existing signal.

**CRITICAL**: DO NOT invent observations — only record what is directly supported by the ground truth above.

⚠ PRE-ANALYSIS (complete both steps in your internal reasoning before writing any entries):
1. SELF-SIGNAL AUDIT: Go through each window summary and extract: (a) which self-LTM signals fired this game, (b) whether the agent successfully followed the corrective Policy (FLAW) or reinforced the tactic (STRENGTH), (c) what the board outcome was. If a FLAW signal fired and the agent repeated the bad habit, prioritize a [MODIFY] on that signal's Policy to make the correction more explicit or forceful. If a FLAW signal fired and the agent successfully followed its corrective Policy, you are STRICTLY FORBIDDEN from proposing a new STRENGTH [ADD] for this success. Instead, rely on the existing FLAW to guide future play. If a STRENGTH signal fired and the agent failed to use the tactic, propose a [MODIFY] to make the reinforcement more explicit. If the agent won and a self-signal's Policy was directly executed and contributed to the win, consider emitting [KEEP] for that signal.
2. CHAT ANALYSIS (only if a chat transcript is present in the game history): Evaluate whether the agent's chat strategy was effective or counterproductive. Note any self-patterns in how the agent used chat.


Each entry in the self-gradient report MUST adhere to these structural rules:

- [REMOVE] Signal: <exact name from database>
  - Reason: <one or two sentences citing the specific ground truth observation that contradicts this signal>

- [ADD] Signal: <new signal name>
  - Reason: <one or two sentences explaining why this new self-pattern is warranted and not covered by any existing entry>
  - Type: <FLAW | STRENGTH>
  - When: <specific trigger condition — max 4 sentences>
  - What: <specific behavior observation — max 4 sentences>
  - Policy: <concrete executable action — max 4 sentences>

- [MODIFY] Signal: <exact name from database>
  - Reason: <one or two sentences explaining what evidence from this game justifies the change>
  - Field: <Name of Field to Change, e.g., When, What, Type, or Policy>
    - Old: <current text>
    - New: <replacement text>
  (List only the fields that are changing. Omit unchanged fields.)

- [MERGE] Signals: <Signal A Name> + <Signal B Name>
  - Reason: <one or two sentences explaining why a single unified policy serves both triggering situations equally well>
  - Into Signal: <new unified signal name>
  - Type: <FLAW | STRENGTH>
  - When: <unified trigger condition — max 4 sentences>
  - What: <unified behavior description — max 4 sentences>
  - Policy: <concrete executable unified policy — max 4 sentences>

- [KEEP] Signal: <exact name from database>
  - Reason: <one or two sentences explaining how this signal's Policy was executed this game and why it was causally beneficial to the agent's win>

⚠ ANTI-VAGUENESS RULE: The Policy MUST name a concrete, executable action. Reject any policy that could apply generically to any game situation.
⚠ BREVITY RULE: Each of When, What, and Policy MUST be at most 4 sentences.

If no notable self-patterns were observed, write: "No signals observed."
Do NOT rewrite the current Self-Reputation Database. Only produce the self-gradient report.
"""


SELF_TGD_SYNTHESIS_PROMPT = """\
You are an AI Memory Optimizer. Your task is to update the Self-Reputation Database for the agent's own play patterns.
You have just finished {n} game(s). Each self-gradient report contains feedback tags:
- [REMOVE]: Signals that have been invalidated by the ground truth.
- [ADD]: New self-patterns to add to the database.
- [MODIFY]: Existing signals with fields that need specific updating.
- [MERGE]: Signals that should be combined into a single new signal.

--- CURRENT SELF-REPUTATION DATABASE ---
{current_self_ltm}

--- SELF-GRADIENT REPORTS ({n} game(s)) ---
{gradient_reports}

--- SELF-LTM FIELD DEFINITIONS ---
* Signal: A short name for the behavioral pattern in the agent's own play.
* Type: FLAW (a recurring mistake to correct) or STRENGTH (an effective tactic to reinforce).
* When: The specific trigger condition — the board state or game context that causes the pattern.
* What: For FLAW: what the agent does incorrectly. For STRENGTH: what the agent does effectively.
* Policy: For FLAW: the corrective action to execute. For STRENGTH: confirmation to reinforce the tactic.
  - The Policy MUST be a concrete, executable action.

--- APPLICATION RULES ---
Your role is Synthesizer. Update the Self-Reputation Database by applying the gradient report(s).
Note: each gradient entry includes a Reason field for your context. Use the Reason to better understand the intent and evidence behind an instruction, but do not copy Reason fields into the final database output.

1. **[REMOVE]**: Find the named signal. Delete it entirely.
2. **[ADD]**: Insert the new signal exactly as written. No changes.
3. **[MODIFY]**: Find the named signal. For each listed field, overwrite the `Old` value with the `New` value. Leave all other fields untouched.
4. **[MERGE]**: Remove both named signals. Insert the merged signal exactly as written.
5. **[KEEP]**: Record that the named signal's Policy was vouched for as causally beneficial in a winning game. The Policy field of this signal is protected — see reconciliation rule 8 below for how to apply this when conflicts arise.
6. **ANTI-VAGUENESS RULE**: The Policy MUST name a concrete, executable action.

--- BATCH RECONCILIATION RULES (apply when {n} > 1) ---
When the same signal receives conflicting instructions across the {n} gradient reports, resolve as follows:

1. **[REMOVE] vs [MODIFY]** on the same signal → keep the signal and apply the [MODIFY].
2. **[ADD] in only 1 game** → include it; if an existing signal describes nearly identical behavior, merge them.
3. **[ADD] in multiple games** → synthesize clusters of the same pattern into one generalized entry; distinct behaviors become separate signals.
4. **[MODIFY] on the same field, consistent direction** → merge into a single coherent New value.
5. **[MODIFY] on the same field, contradicting directions** → take the union; abstract to the more general condition covering both.
6. **[MERGE] where a named signal does not exist** → convert to [MODIFY] on the surviving signal.
7. **[MERGE] in some games, [MODIFY] in others on the same signals** → execute the [MERGE] and fold in the modifications.
8. **[KEEP] vs [MODIFY] on the Policy field of the same signal** → apply the [MODIFY] only if it makes the Policy more specific or correct without changing the core prescribed action. If the [MODIFY] would change the core executable action itself (i.e., prescribe a fundamentally different response), prefer [KEEP] and discard that [MODIFY] on the Policy field. [KEEP] does NOT block [MODIFY] on the When or What fields — those may be updated freely.

--- SYNTHESIS QUALITY RULES ---
- **Abstract and Generalize**: Do not include game-specific details. Produce generalized descriptions that apply across games.
- **Brevity**: Each of When, What, and Policy MUST be at most 4 sentences in the final database. If applying a [MODIFY] or [MERGE] would push a field beyond 4 sentences, distill it — generalize rather than enumerate.
- **Merge-First Discipline**: Before finalizing the output, scan the *entire* resulting database for signals that describe variations of the same underlying self-pattern. Two signals should be merged into one if a single unified policy would serve the agent equally well in both triggering situations. Do not require identical trigger wording — triggers are equivalent if they describe the same underlying game situation from different angles. Emit a [MERGE] for any such pair you identify, even if the gradient reports did not propose one.
- **FLAW/STRENGTH MERGE RULE**: If a FLAW signal and a STRENGTH signal describe the same underlying game situation (e.g., the STRENGTH is simply the successful execution of the FLAW's corrective policy), you MUST [MERGE] them. The resulting merged signal MUST be typed as a FLAW. Frame the new 'What' field to describe the known pitfall or bad habit to avoid, and frame the new 'Policy' field as the proven, successful action to take instead.
- **ANTI-DUPLICATION RULE**: You are STRICTLY FORBIDDEN from finalizing an [ADD] if the core concept is already represented in the database. You MUST instead use [MODIFY] or [MERGE] to elevate the existing signal to a higher level of abstraction that covers the new edge case.
- **Distinct signals only**: Only retain a signal as a separate entry if its trigger condition and required response are clearly distinct from every other signal in the database such that merging would weaken the policy.


Each synthesized memory entry MUST use this format:

- Signal: [Short Name of Pattern]
  - Type: [FLAW | STRENGTH]
  - When: [Specific trigger condition — max 4 sentences]
  - What: [Specific behavior observation — max 4 sentences]
  - Policy: [Concrete executable action — max 4 sentences]

Write ONLY the updated self-memory. Do not include any pleasantries or conversational filler.
If no self-memory exists yet and the gradient report contains ADD signals, write a fresh memory from those signals.
"""
