HIVE_MEMORY_NOTICE = """\
=== HIVE MEMORY NOTICE ===
You are operating in Hive memory mode. Both you and your partner maintain separate private memory databases about each other. You will also receive your partner's observations about YOUR OWN behavior below under a separate section.
=== END HIVE MEMORY NOTICE ==="""

HIVE_UPDATE_NOTICE = ""

PARTNER_VIEW_OF_ME_PROMPT = """\
=== YOUR PARTNER'S OBSERVATIONS ABOUT YOUR PLAY ===
The following signals were developed by your PARTNER by observing YOUR behavior in previous games. This is how you appear to them.

CRITICAL PRONOUN TRANSLATION GUIDE:
Because these notes were written by your partner, the perspectives and pronouns are reversed. When reading the entries below, you MUST translate them as follows:
- "The opponent" or "They" -> Refers to YOU.
- "You" or "Your" in the Policy -> Refers to YOUR PARTNER.

{partner_view_text}

=== END PARTNER OBSERVATIONS ==="""
