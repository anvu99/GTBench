import re
from gamingbench.ltm.prompts import GRADIENT_ENGINE_PROMPT

def run_gradient_engine(
    model,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_ltm: str,
    game_history_legend: str
) -> tuple:
    """Runs the TextGrad strategy gradient engine to analyze match performance.
    
    Returns:
        (structural_report, raw_generation) where:
        - structural_report (str): the ADD/MODIFY/REMOVE/MERGE entries, passed to TGD Synthesizer
        - raw_generation (str): the raw generation text from the model including thinking blocks
    """
    prompt = game_intro + "\n\n" + GRADIENT_ENGINE_PROMPT.format(
        game_history=game_history,
        window_summaries=window_summaries,
        current_ltm=current_ltm,
        game_history_legend=game_history_legend
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block
    generations, _, _ = model.query(messages, n=1, stop=None, prompt_type='move')
    raw_generation = generations[0]
    structural_report = strip_thinking_block(raw_generation).strip()

    return structural_report, raw_generation
