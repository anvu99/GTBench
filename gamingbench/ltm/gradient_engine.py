from gamingbench.ltm.prompts import GRADIENT_ENGINE_PROMPT

def run_gradient_engine(
    model,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_ltm: str
) -> str:
    """Runs the TextGrad strategy gradient engine to analyze match performance."""
    prompt = game_intro + "\n\n" + GRADIENT_ENGINE_PROMPT.format(
        game_history=game_history,
        window_summaries=window_summaries,
        current_ltm=current_ltm
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    from gamingbench.utils.utils import strip_thinking_block
    generations, _, _ = model.query(messages, n=1, stop=None, prompt_type='move')
    return strip_thinking_block(generations[0])
