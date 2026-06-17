from gamingbench.ltm.prompts import TGD_SYNTHESIS_PROMPT

def run_tgd_synthesis(
    model,
    game_intro: str,
    current_ltm: str,
    gradient_report: str
) -> str:
    """Runs the Textual Gradient Descent (TGD) synthesizer to update the LTM."""
    prompt = game_intro + "\n\n" + TGD_SYNTHESIS_PROMPT.format(
        current_ltm=current_ltm,
        gradient_report=gradient_report
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    from gamingbench.utils.utils import strip_thinking_block
    generations, _, _ = model.query(messages, n=1, stop=None, prompt_type='move')
    return strip_thinking_block(generations[0])
