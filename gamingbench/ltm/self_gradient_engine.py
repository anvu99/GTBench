from gamingbench.ltm.prompts import SELF_GRADIENT_ENGINE_PROMPT


def run_self_gradient_engine(
    model,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_self_ltm: str,
    game_history_legend: str
) -> str:
    """Runs the self-gradient engine to analyze the agent's own play patterns.

    Unlike the opponent gradient engine, this returns ONLY the structural report
    (ADD/MODIFY/REMOVE/MERGE entries). There is no Correctness Scores section and
    no EMA scoring — the gradient engine's explicit [REMOVE] tag is the sole
    pruning mechanism for self-LTM signals.

    Returns:
        structural_report (str): The ADD/MODIFY/REMOVE/MERGE entries for the
        Self-TGD Synthesizer.
    """
    prompt = game_intro + "\n\n" + SELF_GRADIENT_ENGINE_PROMPT.format(
        game_history=game_history,
        window_summaries=window_summaries,
        current_self_ltm=current_self_ltm,
        game_history_legend=game_history_legend
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block
    generations, _, _ = model.query(messages, n=1, stop=None, prompt_type='move')
    return strip_thinking_block(generations[0])
