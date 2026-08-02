class ChatChannel:
    def __init__(self, window_size=4):
        self.transcript = []
        self.window_size = window_size

    def reset(self):
        self.transcript = []

    def add_message(self, speaker_idx: int, message: str, round_idx: int = 1):
        self.transcript.append({"round": round_idx, "speaker": speaker_idx, "message": message.strip()})

    def _format_messages(self, messages: list, observer_idx: int) -> str:
        if not messages:
            return "No messages yet."
            
        grouped = {}
        for msg in messages:
            r = msg.get("round", 1)
            if r not in grouped:
                grouped[r] = []
            grouped[r].append(msg)
            
        lines = []
        for r in sorted(grouped.keys()):
            lines.append(f"Round {r}")
            for msg in grouped[r]:
                name = "You" if msg['speaker'] == observer_idx else "Opponent"
                lines.append(f"{name}: {msg['message']}")
            lines.append("") # Empty line between rounds
            
        return "\n".join(lines).strip()

    def get_recent_window(self, observer_idx: int) -> str:
        if not self.transcript:
            return "No messages yet."
        # A pair is 2 messages. Window size K means 2*K messages.
        recent = self.transcript[-(self.window_size * 2):]
        return self._format_messages(recent, observer_idx)

    def get_full_transcript(self, observer_idx: int) -> str:
        return self._format_messages(self.transcript, observer_idx)
