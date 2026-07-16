from gamingbench.ltm.prompts import PROACTIVE_GRADIENT_ENGINE_PROMPT


def run_proactive_gradient_engine(
    model,
    agent_id: str,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_proactive_ltm: str,
    game_history_legend: str
) -> str:
    """Runs the proactive-gradient engine to analyze the agent's own play patterns.

    Unlike the opponent gradient engine, this returns ONLY the structural report
    (ADD/MODIFY/REMOVE/MERGE entries). There is no EMA scoring — the gradient
    engine's explicit [REMOVE] tag is the sole
    pruning mechanism for proactive-LTM signals.

    Returns:
        structural_report (str): The ADD/MODIFY/REMOVE/MERGE entries for the
        Proactive-TGD Synthesizer.
    """
    prompt = game_intro + "\n\n" + PROACTIVE_GRADIENT_ENGINE_PROMPT.format(
        agent_id=agent_id,
        game_history=game_history,
        window_summaries=window_summaries,
        current_proactive_ltm=current_proactive_ltm,
        game_history_legend=game_history_legend
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    structural_report = strip_thinking_block(raw_generation).strip()
    
    if not structural_report:
        structural_report = "No signals observed."
        
    return structural_report, raw_generation, prompt
