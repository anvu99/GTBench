from gamingbench.ltm.prompts import TGD_SYNTHESIS_PROMPT, SEPARATE_TGD_SYNTHESIS_PROMPT


def _format_gradient_reports(gradient_reports: list) -> str:
    """Format a list of structural gradient reports into the numbered block used by TGD_SYNTHESIS_PROMPT."""
    if len(gradient_reports) == 1:
        return f"=== GAME 1 GRADIENT REPORT ===\n{gradient_reports[0]}"
    return "\n\n".join(
        f"=== GAME {i + 1} GRADIENT REPORT ===\n{report}"
        for i, report in enumerate(gradient_reports)
    )


def run_tgd_synthesis(
    model,
    game_intro: str,
    current_ltm: str,
    gradient_reports: list,
) -> str:
    """Runs the Textual Gradient Descent (TGD) synthesizer to update the LTM.

    Args:
        model: The LLM model to use.
        game_intro: Game introduction / rules context.
        current_ltm: The current LTM text before this update.
        gradient_reports: List of structural gradient reports (one per game in the batch).
                          Pass a single-element list for the non-batch (single-game) case.

    Returns:
        The updated LTM text.
    """
    n = len(gradient_reports)
    formatted_reports = _format_gradient_reports(gradient_reports)

    prompt = game_intro + "\n\n" + TGD_SYNTHESIS_PROMPT.format(
        current_ltm=current_ltm,
        n=n,
        gradient_reports=formatted_reports,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    new_ltm = strip_thinking_block(raw_generation)
    
    if not new_ltm.strip():
        # Keep current state if validation and retries failed
        return current_ltm, raw_generation, prompt
        
    return new_ltm, raw_generation, prompt

def run_separate_tgd_synthesis(
    model,
    peer_id: str,
    game_intro: str,
    current_ltm: str,
    gradient_reports: list,
) -> str:
    n = len(gradient_reports)
    formatted_reports = _format_gradient_reports(gradient_reports)

    prompt = game_intro + "\n\n" + SEPARATE_TGD_SYNTHESIS_PROMPT.format(
        peer_id=peer_id,
        current_ltm=current_ltm,
        n=n,
        gradient_reports=formatted_reports,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    new_ltm = strip_thinking_block(raw_generation)
    
    if not new_ltm.strip():
        # Keep current state if validation and retries failed
        return current_ltm, raw_generation, prompt
        
    return new_ltm, raw_generation, prompt
