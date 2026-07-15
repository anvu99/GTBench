from gamingbench.ltm.prompts import SELF_TGD_SYNTHESIS_PROMPT


def _format_self_gradient_reports(gradient_reports: list) -> str:
    """Format a list of self-gradient reports into the numbered block used by SELF_TGD_SYNTHESIS_PROMPT."""
    if len(gradient_reports) == 1:
        return f"=== GAME 1 SELF-GRADIENT REPORT ===\n{gradient_reports[0]}"
    return "\n\n".join(
        f"=== GAME {i + 1} SELF-GRADIENT REPORT ===\n{report}"
        for i, report in enumerate(gradient_reports)
    )


def run_self_tgd_synthesis(
    model,
    game_intro: str,
    current_self_ltm: str,
    gradient_reports: list,
) -> str:
    """Runs the Self-TGD Synthesizer to update the Self-Reputation Database.

    No EMA or auto-removal is performed here — the result is saved directly.

    Args:
        model: The LLM model to use.
        game_intro: Game introduction / rules context.
        current_self_ltm: The current self-LTM text before this update.
        gradient_reports: List of structural self-gradient reports (one per game in the batch).

    Returns:
        The updated self-LTM text.
    """
    n = len(gradient_reports)
    formatted_reports = _format_self_gradient_reports(gradient_reports)

    prompt = game_intro + "\n\n" + SELF_TGD_SYNTHESIS_PROMPT.format(
        current_self_ltm=current_self_ltm,
        n=n,
        gradient_reports=formatted_reports,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    new_self_ltm = strip_thinking_block(raw_generation)
    
    if not new_self_ltm.strip():
        # Keep current state if validation and retries failed
        return current_self_ltm, raw_generation, prompt
        
    return new_self_ltm, raw_generation, prompt
