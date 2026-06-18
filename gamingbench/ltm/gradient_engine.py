import re
from gamingbench.ltm.prompts import GRADIENT_ENGINE_PROMPT

# Mapping from discrete label strings to float scores
SCORE_LABELS: dict = {
    "CONFIRMED": 1.0,
    "MOSTLY_CONFIRMED": 0.75,
    "ABSENT": 0.5,
    "PARTIALLY_CONTRADICTED": 0.25,
    "CONTRADICTED": 0.0,
}

def _split_gradient_report(raw_report: str) -> tuple:
    """Split the raw gradient report into (structural_text, scores_section_text).
    
    The structural text goes to the TGD Synthesizer unchanged.
    The scores section is parsed separately by _parse_correctness_scores().
    """
    if "### Correctness Scores" in raw_report:
        idx = raw_report.index("### Correctness Scores")
        return raw_report[:idx].strip(), raw_report[idx:]
    return raw_report.strip(), ""


def _parse_correctness_scores(scores_section: str) -> dict:
    """Parse '- Signal: <name> → <LABEL>' lines into {signal_name: float}."""
    scores = {}
    label_pattern = "|".join(SCORE_LABELS.keys())
    for match in re.finditer(
        rf"- Signal:\s*(.+?)\s*→\s*({label_pattern})",
        scores_section
    ):
        signal_name = match.group(1).strip()
        label = match.group(2).strip()
        scores[signal_name] = SCORE_LABELS[label]
    return scores


def run_gradient_engine(
    model,
    game_intro: str,
    game_history: str,
    window_summaries: str,
    current_ltm: str
) -> tuple:
    """Runs the TextGrad strategy gradient engine to analyze match performance.
    
    Returns:
        (structural_report, correctness_scores) where:
        - structural_report (str): the ADD/MODIFY/REMOVE/MERGE entries, passed to TGD Synthesizer
        - correctness_scores (dict): {signal_name -> float} for EMA update, parsed from
          the '### Correctness Scores' section of the model's output
    """
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
    raw = strip_thinking_block(generations[0])

    structural_report, scores_section = _split_gradient_report(raw)
    correctness_scores = _parse_correctness_scores(scores_section)

    return structural_report, correctness_scores
