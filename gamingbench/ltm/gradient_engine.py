import re
from gamingbench.ltm.prompts import GRADIENT_ENGINE_PROMPT, SEPARATE_GRADIENT_ENGINE_PROMPT

def run_gradient_engine(
    model,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_ltm: str,
    game_history_legend: str,
    prompt_template: str = GRADIENT_ENGINE_PROMPT
) -> tuple:
    """
    Runs the TextGrad strategy gradient engine to analyze match performance (Opponent Focus).
    This function prompts the LLM to reflect on the opponent's behavior in the recently concluded game,
    comparing it against the `current_ltm` to propose structured updates ([ADD], [MODIFY], [KEEP], etc.).
    
    Returns:
        (structural_report, raw_generation, prompt) where:
        - structural_report (str): the ADD/MODIFY/REMOVE/MERGE entries, passed to TGD Synthesizer
        - raw_generation (str): the raw generation text from the model including <think> blocks
        - prompt (str): the full prompt sent to the model for debugging
    """
    # Format the prompt using the injected game variables
    prompt = game_intro + "\n\n" + prompt_template.format(
        game_history=game_history,
        window_summaries=window_summaries,
        current_ltm=current_ltm,
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

def run_separate_gradient_engine(
    model,
    peer_id: str,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_ltm: str,
    game_history_legend: str,
    prompt_template: str = SEPARATE_GRADIENT_ENGINE_PROMPT
) -> tuple:
    """
    Identical to `run_gradient_engine`, but used in multi-agent or hive scenarios 
    where we explicitly track reputations per specific `peer_id`.
    """
    # Format the prompt, substituting the specific peer's ID
    prompt = game_intro + "\n\n" + prompt_template.format(
        peer_id=peer_id,
        game_history=game_history,
        window_summaries=window_summaries,
        current_ltm=current_ltm,
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
