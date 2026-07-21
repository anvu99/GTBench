from gamingbench.ltm.prompts import SELF_GRADIENT_ENGINE_PROMPT


def run_self_gradient_engine(
    model,
    agent_id: str,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_self_ltm: str,
    game_history_legend: str,
    prompt_template: str = SELF_GRADIENT_ENGINE_PROMPT
) -> str:
    """Runs the self-gradient engine to analyze the agent's own play patterns.

    Unlike the opponent gradient engine, which looks for exploits, the self-gradient engine
    looks for repeated mistakes, vulnerabilities, or personal flaws that the agent exhibited
    in the game. It uses explicit [REMOVE] tags to prune bad habits.

    Returns:
        (structural_report, raw_generation, prompt) where:
        - structural_report (str): the ADD/MODIFY/REMOVE/MERGE entries, passed to TGD Synthesizer
        - raw_generation (str): the raw generation text from the model including <think> blocks
        - prompt (str): the full prompt sent to the model for debugging
    """
    # Inject the agent's ID into the prompt so the LLM knows which side to critique
    prompt = game_intro + "\n\n" + prompt_template.format(
        agent_id=agent_id,
        game_history=game_history,
        window_summaries=window_summaries,
        current_self_ltm=current_self_ltm,
        game_history_legend=game_history_legend
    )

    messages = [
        {"role": "user", "content": prompt}
    ]

    from gamingbench.utils.utils import strip_thinking_block, query_with_thinking_validation
    
    # Query the LLM, validating that it correctly wrapped its reasoning in <think> tags
    raw_generation = query_with_thinking_validation(model, messages, prompt_type='move')
    
    # Strip the <think> blocks to isolate the pure structural instructions for the Synthesis Engine
    structural_report = strip_thinking_block(raw_generation).strip()
    
    if not structural_report:
        structural_report = "No signals observed."
        
    return structural_report, raw_generation, prompt
