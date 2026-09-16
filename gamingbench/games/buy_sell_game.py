import re
import copy
import logging
from gamingbench.utils.history_tracker import GameMatch, Step

logger = logging.getLogger(__name__)

SELLER_IDX = 0
BUYER_IDX = 1

COST_OF_PRODUCTION = 40
WILLINGNESS_TO_PAY = 60


class BuySellGame:
    """
    A bilateral negotiation game (Buyer-Seller) ported from NegotiationArena.

    Roles:
      - Player 0 = Seller: has 1 unit of item X, private cost_of_production.
      - Player 1 = Buyer : has money, private willingness_to_pay.

    Turn structure:
      - Alternating turns, Seller goes first.
      - Max rounds configurable via YAML (default 10).

    Actions:
      - <PROPOSE: X>  – propose a price of $X (integer).
      - <ACCEPT>      – accept the most recent proposal. Game ends.
      - <REJECT>      – reject and end negotiations. Both score 0.

    Scoring on ACCEPT:
      - Seller score = agreed_price - cost_of_production
      - Buyer  score = willingness_to_pay - agreed_price
      - No deal      = 0 for both.
    """

    def __init__(self, config=None) -> None:
        self.game_name = "buy_sell_game"
        self.game = self
        self.config = config
        self.status = "Normal"

        # Read values from YAML config; fall back to NegotiationArena defaults.
        self.cost = int(getattr(config, "cost_of_production", COST_OF_PRODUCTION))
        self.wtp = int(getattr(config, "willingness_to_pay", WILLINGNESS_TO_PAY))
        self.max_rounds = int(getattr(config, "max_rounds", 10))

    def reset(self) -> None:
        self.status = "Normal"

    # ──────────────────────────────────────────────────────────────────────────
    # Main game loop
    # ──────────────────────────────────────────────────────────────────────────

    def play(self, agent_list, model_list, tracker, **kwargs) -> None:
        self.status = "Normal"
        match = GameMatch()

        # Private valuations per player
        private_val = {SELLER_IDX: self.cost, BUYER_IDX: self.wtp}
        roles = {SELLER_IDX: "Seller", BUYER_IDX: "Buyer"}

        last_proposed_price: int | None = None
        last_proposer_idx: int | None = None
        game_over = False
        final_scores = {SELLER_IDX: 0.0, BUYER_IDX: 0.0}
        winner_idx = None

        # We record every turn for post_game_update history reconstruction.
        steps_record = []  # list of (player_idx, board_str, action_str)

        # ── [LTM] Initialise agent game state ─────────────────────────────────
        from gamingbench.prompts.observation_prompts import construct_game_intro
        game_intro = construct_game_intro(self.game_name)
        for i, agent in enumerate(agent_list):
            # Always set current_game_intro so SW/Tendency agents have context
            # for their memory update prompts (flush_batch_updates uses this field).
            agent.current_game_intro = game_intro
            agent.current_game_name = self.game_name
            if hasattr(agent, "reset_game_state"):
                opp_idx = 1 - i
                opp_agent = agent_list[opp_idx]
                if (hasattr(agent, "agent_name")
                        and hasattr(opp_agent, "agent_name")
                        and agent.agent_name == opp_agent.agent_name):
                    opp_name = opp_agent.agent_name
                else:
                    opp_name = (
                        f"{opp_agent.agent_name}_{model_list[opp_idx].nick_name}"
                    )
                agent.reset_game_state(opp_name, game_intro)

        # ── Main loop ─────────────────────────────────────────────────────────
        for turn in range(1, self.max_rounds + 1):
            if game_over:
                break

            # Seller = turn 1, 3, 5 … ; Buyer = turn 2, 4, 6 …
            player_idx = SELLER_IDX if turn % 2 == 1 else BUYER_IDX
            agent = agent_list[player_idx]
            model = model_list[player_idx]

            # Build per-turn board string
            board_str = self._build_board_str(
                player_idx, turn, last_proposed_price, last_proposer_idx, roles
            )

            # Legal moves
            legal_moves = self._get_legal_moves(last_proposed_price, player_idx, last_proposer_idx)

            obs_dict = {
                "env_name": self.game_name,
                "player_idx": player_idx,
                "player_role": roles[player_idx],
                "private_valuation": private_val[player_idx],
                "last_proposed_price": last_proposed_price,
                "last_proposer_role": roles[last_proposer_idx] if last_proposer_idx is not None else None,
                "turn_number": turn,
                "max_turns": self.max_rounds,
                "board": board_str,
                "legal_moves": legal_moves,
            }

            # ── Step ──────────────────────────────────────────────────────────
            _step = Step(agent.agent_name)
            _step.set_model_name(model.nick_name)
            _step.set_observation(copy.deepcopy(obs_dict))

            action, query_list = agent.step(obs_dict)
            for q in query_list:
                _step.add_query(q)
            _step.set_move(action)
            match.add_step(_step)

            steps_record.append((player_idx, board_str, action))
            logger.info(
                f"Turn {turn} | Player {player_idx} ({roles[player_idx]}) | action: {action}"
            )

            # ── Parse action ──────────────────────────────────────────────────
            action_type, price = self._parse_action(action)

            if action_type is None:
                # Invalid move – treat as abnormal
                logger.warning(
                    f"Player {player_idx} ({roles[player_idx]}) made an invalid move: {action!r}"
                )
                match.status = "Abnormal"
                self.status = "Abnormal"
                match.agents_at_fault.append(agent.agent_name)
                game_over = True

            elif action_type == "REJECT":
                game_over = True
                # Scores stay 0

            elif action_type == "ACCEPT":
                if last_proposed_price is None:
                    # Cannot accept when there's no proposal yet — treat as invalid
                    logger.warning(
                        f"Player {player_idx} tried to ACCEPT with no prior proposal. Treating as REJECT."
                    )
                    game_over = True
                else:
                    agreed_price = last_proposed_price
                    final_scores[SELLER_IDX] = float(agreed_price - self.cost)
                    final_scores[BUYER_IDX] = float(self.wtp - agreed_price)
                    if final_scores[SELLER_IDX] > final_scores[BUYER_IDX]:
                        winner_idx = SELLER_IDX
                    elif final_scores[BUYER_IDX] > final_scores[SELLER_IDX]:
                        winner_idx = BUYER_IDX
                    # else: equal surplus → draw
                    game_over = True

            elif action_type == "PROPOSE":
                last_proposed_price = price
                last_proposer_idx = player_idx

            # If we've used all rounds without a deal, game ends with 0 scores.

        # ── Record scores ─────────────────────────────────────────────────────
        s0, s1 = final_scores[SELLER_IDX], final_scores[BUYER_IDX]
        if winner_idx == SELLER_IDX:
            match.set_winner(
                agent_list[SELLER_IDX].agent_name + "_" + model_list[SELLER_IDX].nick_name
            )
            match.winner_score = s0
            match.loser_score = s1
        elif winner_idx == BUYER_IDX:
            match.set_winner(
                agent_list[BUYER_IDX].agent_name + "_" + model_list[BUYER_IDX].nick_name
            )
            match.winner_score = s1
            match.loser_score = s0
        else:
            match.set_winner("")
            match.winner_score = max(s0, s1)
            match.loser_score = min(s0, s1)

        tracker.add_match(match)

        if not self.is_match_normal():
            logger.info("Match ended abnormally. Skipping post_game_update.")
            return

        # ── post_game_update for memory agents ────────────────────────────────
        results = [s0, s1]
        for agent_idx, agent in enumerate(agent_list):
            if hasattr(agent, "post_game_update"):
                agent_history = self._build_agent_history(
                    agent_idx, steps_record, results, roles
                )
                final_board = self._build_final_board_str(results, roles)
                try:
                    agent.post_game_update(
                        agent_history,
                        final_board_state=final_board,
                        env_name=self.game_name,
                    )
                except TypeError:
                    try:
                        agent.post_game_update(agent_history, final_board_state=final_board)
                    except TypeError:
                        agent.post_game_update(agent_history)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_action(self, action: str):
        """
        Returns ('PROPOSE', price) | ('ACCEPT', None) | ('REJECT', None) | (None, None).
        Case-insensitive.
        """
        action = action.strip()
        if re.search(r'<\s*ACCEPT\s*>', action, re.IGNORECASE):
            return ('ACCEPT', None)
        if re.search(r'<\s*REJECT\s*>', action, re.IGNORECASE):
            return ('REJECT', None)
        m = re.search(r'<\s*PROPOSE\s*:\s*(\d+)\s*>', action, re.IGNORECASE)
        if m:
            return ('PROPOSE', int(m.group(1)))
        return (None, None)

    def _get_legal_moves(self, last_price, current_player_idx, last_proposer_idx):
        """Build the legal_moves list shown to the agent.
        
        We return [] here to bypass the strict string matching in PromptAgent.step().
        Since PROPOSE takes any integer, we cannot list all valid strings. 
        The human-readable instructions are hardcoded in the observation prompt instead.
        """
        return []

    def _build_board_str(
        self, player_idx, turn, last_price, last_proposer_idx, roles
    ) -> str:
        role = roles[player_idx]
        if player_idx == SELLER_IDX:
            val_label = f"Your cost of production: {self.cost}"
        else:
            val_label = f"Your willingness to pay: {self.wtp}"

        if last_price is None:
            offer_str = "None (no offer yet)"
        else:
            proposer_role = roles[last_proposer_idx]
            offer_str = f"${last_price} (proposed by {proposer_role})"

        return (
            f"Role={role}, {val_label}, "
            f"Turn={turn}/{self.max_rounds}, "
            f"Last offer={offer_str}"
        )

    def _build_final_board_str(self, results, roles) -> str:
        s0, s1 = results[SELLER_IDX], results[BUYER_IDX]
        return (
            f"Game over. Seller score={s0}, Buyer score={s1}."
        )

    def _build_agent_history(
        self, agent_idx: int, steps_record: list, results: list, roles: dict
    ) -> str:
        """Build the history string passed to agent.post_game_update()."""
        from gamingbench.prompts.observation_prompts import buy_sell_game as bsg_prompts
        legend = bsg_prompts._construct_game_history_legend()
        history = legend

        for step_idx, (p_idx, board_str, action_str) in enumerate(steps_record):
            current_round = step_idx + 1
            is_self = (p_idx == agent_idx)
            prefix = "You" if is_self else "Opponent"

            if is_self:
                history += f"Round {current_round} (Your move):\n"
            else:
                history += f"Round {current_round} (Opponent's move):\n"

            history += f"  [Position]: {board_str}\n"
            history += f"  [Move] {prefix}: {action_str}\n\n"

        your_score = results[agent_idx]
        opp_score = results[1 - agent_idx]
        history += f"Game Outcome: Your score={your_score}, Opponent score={opp_score}"
        return history

    def is_match_normal(self) -> bool:
        return self.status == "Normal"
