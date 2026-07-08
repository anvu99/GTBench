from gamingbench.games.negotiation import Negotiation

class CooperativeNegotiation(Negotiation):
    def __init__(self) -> None:
        super().__init__()
        # Override the game name so GTBench uses the cooperative observation prompts
        self.game_name = "cooperative_negotiation"
