from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable

from dedalus_labs import AsyncDedalus

from llm_arena.core.player import LLMPlayer
from llm_arena.core.types import (
    ActionResult,
    GameConfig,
    GameOutcome,
    GamePhase,
    PhaseType,
    PlayerRole,
)
from llm_arena.logging.transcript import TranscriptLogger


def _serialize_message(msg: Any) -> dict:
    """Convert a Dedalus Message object to a plain dict for transcript storage."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "__dict__"):
        return {k: v for k, v in msg.__dict__.items() if not k.startswith("_")}
    return {"content": str(msg)}


def _extract_assistant_reasoning(messages: list) -> str | None:
    """Pull reasoning text from assistant messages when final_output is empty."""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            msg = _serialize_message(msg)
        if msg.get("role") == "assistant" and msg.get("content"):
            parts.append(msg["content"])
    return "\n".join(parts) if parts else None


def _attach_llm_reasoning(processed: "ActionResult", run_result: Any) -> None:
    """Attach LLM reasoning from a Dedalus RunResult onto an ActionResult."""
    processed.llm_output = getattr(run_result, "final_output", None)
    raw_msgs = getattr(run_result, "messages", None)
    if raw_msgs:
        serialized = [_serialize_message(m) for m in raw_msgs]
        processed.llm_messages = serialized
        if not processed.llm_output:
            processed.llm_output = _extract_assistant_reasoning(serialized)


class BaseGame(ABC):
    """
    Abstract base class for all games in LLMArena.

    Subclasses must implement:
      - setup()               : initialize game state, assign roles
      - get_next_phase()      : determine what phase comes next
      - get_player_view()     : build the prompt a player sees
      - get_tools_for_player(): return tool functions a player can use now
      - process_action()      : validate and apply a player's action
      - check_game_over()     : return GameOutcome if finished, else None
    """

    default_players: int = 2  # override in subclass

    def __init__(self, config: GameConfig):
        self.config = config
        self.game_id: str = str(uuid.uuid4())[:8]
        self.client = AsyncDedalus()
        self.players: dict[str, LLMPlayer] = {}
        self.roles: dict[str, PlayerRole] = {}
        self.state: dict[str, Any] = {}
        self.phase: GamePhase | None = None
        self.action_log: list[ActionResult] = []
        self.transcript = TranscriptLogger()
        self.round_number: int = 0

    MAX_CONSECUTIVE_FAILURES: int = 5

    async def run(self) -> GameOutcome:
        """Main game loop."""
        await self.setup()
        self.transcript.log_game_start(self.game_id, self.config)
        prev_action_count = 0
        consecutive_no_progress = 0

        while True:
            self.phase = await self.get_next_phase()
            self.transcript.log_phase(self.phase)

            if self.phase.phase_type == PhaseType.GAME_OVER:
                break

            if self.phase.phase_type == PhaseType.DISCUSSION:
                await self._run_discussion_phase()
            else:
                await self._run_action_phase()

            # Detect stuck games (no successful actions advancing state)
            new_count = len(self.action_log)
            has_progress = new_count > prev_action_count and any(
                a.success for a in self.action_log[prev_action_count:]
            )
            if has_progress:
                consecutive_no_progress = 0
            else:
                consecutive_no_progress += 1
            prev_action_count = new_count

            if consecutive_no_progress >= self.MAX_CONSECUTIVE_FAILURES:
                if self.config.verbose:
                    print(f"\nAborting: {consecutive_no_progress} consecutive turns with no progress.")
                break

            outcome = await self.check_game_over()
            if outcome is not None:
                self.transcript.log_game_end(outcome)
                return outcome

        outcome = await self.check_game_over()
        if outcome is None:
            # Game aborted (e.g. stuck detection) — declare a draw
            from datetime import datetime, timezone
            outcome = GameOutcome(
                game_id=self.game_id,
                game_type=self.config.game_type,
                winner_ids=[],
                loser_ids=[],
                metadata={"termination": "aborted"},
                timestamp=datetime.now(timezone.utc),
            )
        self.transcript.log_game_end(outcome)
        return outcome

    async def _run_action_phase(self):
        """Each active player takes a turn via tool calls."""
        for player_id in self.phase.active_player_ids:
            player = self.players[player_id]
            view = await self.get_player_view(player_id)
            tools = await self.get_tools_for_player(player_id)

            if not tools:
                continue

            action_result = await player.take_action(
                game_prompt=view,
                tools=tools,
                phase=self.phase,
            )

            processed = await self.process_action(player_id, action_result)
            _attach_llm_reasoning(processed, action_result)
            self.action_log.append(processed)
            self.transcript.log_action(processed)

            if self.config.verbose:
                print(f"  [{player.info.name}] {processed.action_name}: {processed.result}")

    async def _run_discussion_phase(self):
        """Players speak in turn, each seeing prior statements."""
        discussion_history: list[dict[str, str]] = []

        for player_id in self.phase.active_player_ids:
            player = self.players[player_id]
            view = await self.get_player_view(player_id)

            if discussion_history:
                view += "\n\n## Discussion so far:\n" + "\n".join(
                    f"{d['name']}: {d['statement']}" for d in discussion_history
                )

            tools = await self.get_tools_for_player(player_id)
            if not tools:
                continue

            action_result = await player.take_action(
                game_prompt=view,
                tools=tools,
                phase=self.phase,
            )

            processed = await self.process_action(player_id, action_result)
            _attach_llm_reasoning(processed, action_result)
            self.action_log.append(processed)
            self.transcript.log_action(processed)

            discussion_history.append(
                {"name": player.info.name, "statement": processed.result}
            )

            if self.config.verbose:
                print(f"  [{player.info.name}] {processed.action_name}: {processed.result}")

    @classmethod
    def default_player_count(cls) -> int:
        return cls.default_players

    # --- Abstract methods ---

    @abstractmethod
    async def setup(self) -> None:
        ...

    @abstractmethod
    async def get_next_phase(self) -> GamePhase:
        ...

    @abstractmethod
    async def get_player_view(self, player_id: str) -> str:
        ...

    @abstractmethod
    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        ...

    @abstractmethod
    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        ...

    @abstractmethod
    async def check_game_over(self) -> GameOutcome | None:
        ...
