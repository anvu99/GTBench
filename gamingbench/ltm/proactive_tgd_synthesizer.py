from gamingbench.ltm.prompts import PROACTIVE_TGD_SYNTHESIS_PROMPT


def _format_proactive_gradient_reports(gradient_reports: list) -> str:
    """Format a list of proactive-gradient reports into the numbered block used by PROACTIVE_TGD_SYNTHESIS_PROMPT."""
    if len(gradient_reports) == 1:
        return f"=== GAME 1 PROACTIVE-GRADIENT REPORT ===\n{gradient_reports[0]}"
    return "\n\n".join(
        f"=== GAME {i + 1} PROACTIVE-GRADIENT REPORT ===\n{report}"
        for i, report in enumerate(gradient_reports)
    )


def run_proactive_tgd_synthesis(
    model,
    game_intro: str,
    current_proactive_ltm: str,
    gradient_reports: list,
) -> str:
    """Runs the Proactive-TGD Synthesizer to update the Overall Strategy Database.

    No EMA or auto-removal is performed here — the result is saved directly.

    Args:
        model: The LLM model to use.
        game_intro: Game introduction / rules context.
        current_proactive_ltm: The current proactive-LTM text before this update.
        gradient_reports: List of structural proactive-gradient reports (one per game in the batch).

    Returns:
        The updated proactive-LTM text.
    """
    n = len(gradient_reports)
    formatted_reports = _format_proactive_gradient_reports(gradient_reports)

    prompt = game_intro + "\n\n" + PROACTIVE_TGD_SYNTHESIS_PROMPT.format(
        current_proactive_ltm=current_proactive_ltm,
        n=n,
        gradient_reports=formatted_reports,
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    new_proactive_ltm = strip_thinking_block(raw_generation)
    
    if not new_proactive_ltm.strip():
        # Keep current state if validation and retries failed
        return current_proactive_ltm, raw_generation, prompt
        
    return new_proactive_ltm, raw_generation, prompt
