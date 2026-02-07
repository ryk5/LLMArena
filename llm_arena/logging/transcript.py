from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_arena.core.types import ActionResult, GameConfig, GameOutcome, GamePhase


class TranscriptLogger:
    """Logs full game transcripts as JSON + human-readable text."""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.game_id: str = ""

    def log_game_start(self, game_id: str, config: GameConfig):
        self.game_id = game_id
        self.events.append(
            {
                "event": "game_start",
                "game_id": game_id,
                "game_type": config.game_type,
                "players": [p.model_dump() for p in config.players],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def log_phase(self, phase: GamePhase):
        self.events.append(
            {
                "event": "phase_change",
                "phase": phase.model_dump(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def log_action(self, action: ActionResult):
        self.events.append(
            {
                "event": "action",
                "action": action.model_dump(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def log_game_end(self, outcome: GameOutcome):
        self.events.append(
            {
                "event": "game_end",
                "outcome": outcome.model_dump(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._write()

    def _write(self):
        json_path = self.log_dir / f"{self.game_id}.json"
        with open(json_path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)

        txt_path = self.log_dir / f"{self.game_id}.txt"
        with open(txt_path, "w") as f:
            for event in self.events:
                if event["event"] == "game_start":
                    f.write(f"=== GAME {event['game_id']} ({event['game_type']}) ===\n")
                    for p in event["players"]:
                        f.write(f"  Player: {p['name']} ({p['model']})\n")
                elif event["event"] == "phase_change":
                    phase = event["phase"]
                    f.write(
                        f"\n--- {phase['phase_type']} (Round {phase['round_number']}) ---\n"
                    )
                    f.write(f"    {phase['description']}\n")
                elif event["event"] == "action":
                    a = event["action"]
                    f.write(f"  [{a['player_id']}] {a['action_name']}: {a['result']}\n")
                    if a.get("llm_output"):
                        f.write(f"    Reasoning: {a['llm_output']}\n")
                elif event["event"] == "game_end":
                    o = event["outcome"]
                    f.write("\n=== GAME OVER ===\n")
                    f.write(f"  Winners: {o['winner_ids']}\n")
                    f.write(f"  Losers:  {o['loser_ids']}\n")
