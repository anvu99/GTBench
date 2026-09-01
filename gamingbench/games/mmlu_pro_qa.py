import os
import re
import json
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from gamingbench.utils.history_tracker import GameMatch, Step

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output.

    Handles two formats:
    1. Complete <think>...</think> blocks.
    2. Lone </think> tag (opening tag missing, common in cached responses) —
       strips everything up to and including the closing tag.
    """
    if "</think>" not in text:
        return " "
        
    # Remove complete <think>...</think> blocks first
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # If a bare </think> remains, strip everything before (and including) it
    text = re.sub(r'^.*?</think>\s*', '', text, flags=re.DOTALL)
    return text.strip()

class MmluProQA:
    _dataset = None
    _cache = None
    # Dynamically resolve the GTBench directory relative to this file
    _cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "global_spoke_cache_mmlu_pro.json"
    )
    _experts = [
        "Expert_Math_Tech", 
        "Expert_Natural_Science", 
        "Expert_Medical_Psych", 
        "Expert_Humanities", 
        "Expert_Business"
    ]
    
    _expert_domains = {
        "Expert_Math_Tech": {"math", "computer science", "engineering"},
        "Expert_Natural_Science": {"physics", "chemistry", "biology"},
        "Expert_Medical_Psych": {"health", "psychology"},
        "Expert_Humanities": {"history", "philosophy", "law"},
        "Expert_Business": {"business", "economics", "other"}
    }
    
    _cache_lock = threading.Lock()

    def __init__(self, config=None) -> None:
        self.game_name = "mmlu_pro_qa"
        self.game = self
        self.config = config
        self.status = "Normal"
        self._load_data()

    @classmethod
    def _load_data(cls):
        if cls._dataset is None:
            try:
                from datasets import load_dataset
                dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
                # Sample questions deterministically
                rng = random.Random(42)
                indices = list(range(len(dataset)))
                rng.shuffle(indices)
                
                # We need to know how many matches. Default to 100 or use config if needed
                cls._dataset = []
                for idx in indices:
                    cls._dataset.append(dataset[idx])
                    
            except Exception as e:
                logger.error(f"Failed to load MMLU Pro dataset: {e}")
                cls._dataset = []
                
        if cls._cache is None:
            if os.path.exists(cls._cache_path):
                with open(cls._cache_path, "r") as f:
                    cls._cache = json.load(f)
            else:
                cls._cache = {}

    def _save_cache(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(MmluProQA._cache_path), exist_ok=True)
        tmp_path = MmluProQA._cache_path + f".{os.getpid()}.{threading.get_ident()}.tmp"
        
        with MmluProQA._cache_lock:
            # Prevent dictionary resize during json.dump
            with open(tmp_path, "w") as f:
                json.dump(MmluProQA._cache, f, indent=4)
                
        # Atomic replace to avoid file corruption when multiple SLURM jobs write simultaneously
        os.replace(tmp_path, MmluProQA._cache_path)

    def _generate_spoke_answers(self, model, question_data):
        question_id = str(question_data.get('question_id'))
        
        with MmluProQA._cache_lock:
            if question_id in MmluProQA._cache:
                return MmluProQA._cache[question_id]
            
        logger.info(f"Generating Spoke answers for question {question_id}...")
        
        spoke_answers = {}
        category = question_data.get('category', '').lower()
        cot_content = question_data.get('cot_content', '')
        
        question = question_data.get('question', '')
        options = question_data.get('options', [])
        options_str = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        
        def generate_for_expert(expert):
            # Check if specialization matches
            expert_categories = MmluProQA._expert_domains.get(expert, set())
            
            hint = ""
            if category in expert_categories:
                hint = f"\n\nCRITICAL HINT: This is the exact correct answer and reasoning. You MUST base your response on this:\n{cot_content}"
                
            prompt = f"You are {expert}, a Spoke in a Hub-and-Spoke system. You and your peers will generate reasoning for your Hub to base its final answer on. Your utility is maximized if the Hub successfully answers the question based on your reasoning. If this question falls into your domain, you will receive a critical hint containing the exact correct answer. IMPORTANT: You are NOT allowed to explicitly mention the existence of the hint or state that you were given the answer. Formulate your reasoning as if you derived it independently.\n\nQuestion:\n{question}\n\nOptions:\n{options_str}{hint}\n\nPlease provide your step-by-step reasoning."
            messages = [{"role": "user", "content": prompt}]
            
            responses, _, _ = model.query(messages, n=1, stop=None, prompt_type='move', enable_thinking=True)
            if responses and len(responses) > 0:
                reasoning = responses[0]
            else:
                reasoning = "Failed to generate reasoning."
                
            return expert, {"raw_reasoning": reasoning}
            
        with ThreadPoolExecutor(max_workers=len(MmluProQA._experts)) as executor:
            results = executor.map(generate_for_expert, MmluProQA._experts)
            
        for expert, result in results:
            spoke_answers[expert] = result
            
        with MmluProQA._cache_lock:
            MmluProQA._cache[question_id] = spoke_answers
            
        self._save_cache()
        return spoke_answers

    def play(self, agent_list, model_list, tracker, **kwargs):
        self.status = "Normal"
        match = GameMatch()
        
        agent = agent_list[0]
        model = model_list[0]
        
        # Determine match index to pick the question.
        # match_idx must be passed as a kwarg from the orchestration layer;
        # falling back to 0 would make every game use the same question.
        match_idx = kwargs.get('match_idx', 0)

        if match_idx >= len(MmluProQA._dataset):
            logger.error("Match index exceeds dataset size.")
            return
            
        question_data = MmluProQA._dataset[match_idx]
        
        # 1. Generate or retrieve Spoke answers
        spoke_answers = self._generate_spoke_answers(model, question_data)
        
        # 2. Build observation
        question = question_data.get('question', '')
        category = question_data.get('category', 'Unknown')
        options = question_data.get('options', [])
        options_str = "\n".join([f"{chr(65+i)}: {opt}" for i, opt in enumerate(options)])
        
        board_str = f"Category: {category}\n\nQuestion:\n{question}\n\nOptions:\n{options_str}\n\n"
        for expert in MmluProQA._experts:
            if expert in spoke_answers:
                spoke_idx = MmluProQA._experts.index(expert) + 1
                answer = spoke_answers[expert]
                raw = answer.get('raw_reasoning', '')
                cleaned = _strip_thinking(raw)
                board_str += f"--- Spoke {spoke_idx} ---\n{cleaned}\n\n"
            
        legal_moves = [f"<{chr(65+i)}>" for i in range(len(options))]
        
        observation_dict = {
            'env_name': self.game_name,
            'board': board_str,
            'legal_moves': legal_moves,
            'player_idx': 0
        }
        
        # 3. Get Hub Action
        action, query_list = agent.step(observation_dict)
        
        # 4. Record Step
        step = Step(agent.agent_name)
        step.set_model_name(model.nick_name)
        step.set_observation(observation_dict)
        step.set_move(action)
        for q in query_list:
            step.add_query(q)
            
        match.add_step(step)
        
        # 5. Evaluate
        try:
            correct_index = question_data.get('answer_index')
            if correct_index is None:
                correct_letter = question_data.get('answer')
            else:
                correct_letter = chr(65 + correct_index)
                
            chosen_letter = action.strip("<> \n")
            if chosen_letter.upper() == correct_letter.upper():
                match.winner_score = 1.0
                match.set_winner(agent.agent_name)
                logger.info("Agent answered correctly.")
            else:
                match.winner_score = 0.0
                match.set_winner("")
                logger.info(f"Agent answered incorrectly. Chosen: {chosen_letter}, Correct: {correct_letter}")
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            match.winner_score = 0.0
            
        tracker.add_match(match)
        
        # 6. Post Game Update for LTM
        if hasattr(agent, 'post_game_update'):
            agent_history = "[Position Legend] Each [Position] line shows the question, options, and peer answers.\n" \
                            "Format: Question text followed by peer reasoning.\n" \
                            "[Move] lines show the final action taken by you.\n\n"
            
            agent_history += f"Round 1 (Your move):\n"
            agent_history += f"  [Position]: {board_str}\n"
            agent_history += f"  [Move] You: {action}\n\n"
            agent_history += f"Game Outcome: Your score={match.winner_score}\n"
            
            try:
                agent.post_game_update(agent_history, final_board_state=board_str, env_name=self.game_name)
            except TypeError:
                try:
                    agent.post_game_update(agent_history, final_board_state=board_str)
                except TypeError:
                    agent.post_game_update(agent_history)

    def reset(self):
        self.status = "Normal"
