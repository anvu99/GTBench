SUCCESSFUL_MEMORY_SI = """
You are an expert in game strategy. You will be given the game introduction and the corresponding trajectory that represents **how an agent successfully accomplished the task**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar games.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable strategies over abstract principles. Do not embed specific game states or exact move sequences from the trajectory.

## Output Format
Your output must strictly follow the Markdown format shown below:

# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplish similar games in the future>
"""

FAILED_MEMORY_SI = """
You are an expert in game strategy. You will be given the game introduction and the corresponding trajectory that represents **how an agent attempted to play but failed or underperformed**.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar games.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Prefer concrete, actionable recovery procedures over abstract principles. Do not embed specific game states or exact move sequences from the trajectory.

## Output Format
Your output must strictly follow the Markdown format shown below:

# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary describing when or when NOT to use the memory item>
## Content <1-3 sentences describing the insights learned to avoid such failures and successfully accomplish similar games in the future>
"""

MEMORY_INJECTION_PROMPT = """--- REASONING BANK MEMORIES ---
The following are structured memory items you have learned from past games. Use them as strategic guidance when making decisions:

{memory_text}
--- END OF MEMORIES ---"""
