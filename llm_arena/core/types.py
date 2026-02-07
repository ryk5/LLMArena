from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class PlayerRole(BaseModel):
    """Role assigned to a player in a game."""

    name: str
    team: str
    description: str
    is_hidden: bool = True


class PlayerInfo(BaseModel):
    """Public-facing info about a player."""

    player_id: str
    name: str  # display name like "claude-opus-4-6"
    model: str  # "anthropic/claude-opus-4-6"


class ActionResult(BaseModel):
    """Result of a player taking an action via a tool call."""

    player_id: str
    action_name: str
    action_args: dict[str, Any] = {}
    result: str
    success: bool
    visible_to: list[str] | None = None  # None means all players
    llm_output: str | None = None  # LLM's text reasoning for this turn
    llm_messages: list[dict] | None = None  # full message history for the turn


class PhaseType(str, Enum):
    SETUP = "setup"
    DISCUSSION = "discussion"
    VOTING = "voting"
    ACTION = "action"
    RESOLUTION = "resolution"
    GAME_OVER = "game_over"


class GamePhase(BaseModel):
    """Current phase of the game."""

    phase_type: PhaseType
    round_number: int
    description: str
    active_player_ids: list[str]
    time_limit: float | None = None


class GameOutcome(BaseModel):
    """Final result of a game."""

    game_id: str
    game_type: str
    winner_ids: list[str]
    loser_ids: list[str]
    ranking: list[str] | None = None
    metadata: dict[str, Any] = {}
    timestamp: datetime


class GameConfig(BaseModel):
    """Configuration for running a game."""

    game_type: str
    players: list[PlayerInfo]
    options: dict[str, Any] = {}
    max_rounds: int = 50
    verbose: bool = False
