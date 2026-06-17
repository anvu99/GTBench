CHAT_INSTRUCTION = """\
Before making your next game move, you have the opportunity to send a short chat message to the opponent. 
Use this channel highly strategically to increase your chances of winning. Depending on the game state, you should actively use the chat to:
- Make a deal, negotiate, or collude if it benefits your long-term goal.
- Deceive or bluff to hide your true intentions or misdirect the opponent.
- Intimidate, distract, or influence the opponent's strategy to force a mistake.
- Build false trust to set up a future betrayal.

Write a single, concise chat message to the opponent utilizing these psychological and strategic tactics. 
Do NOT output a game move, and do NOT output internal reasoning. Just output the message text you wish to send.
"""

CHAT_HISTORY_INJECTION = """\
--- ONGOING CHAT ---
{chat_history}
"""
