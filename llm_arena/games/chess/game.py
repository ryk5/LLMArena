from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import chess

from llm_arena.core.game import BaseGame
from llm_arena.core.player import LLMPlayer
from llm_arena.core.types import (
    ActionResult,
    GameOutcome,
    GamePhase,
    PhaseType,
    PlayerRole,
)
from llm_arena.games import register_game
from llm_arena.games.chess.prompts import CHESS_SYSTEM_PROMPT, get_turn_prompt
from llm_arena.games.chess.tools import create_chess_tools


@register_game("chess")
class ChessGame(BaseGame):
    """
    Chess game implementation for the LLM Arena.

    Two LLM players play a full game of chess. The game uses the python-chess
    library for board state management and move validation. Tools are closures
    that directly mutate the board, so by the time process_action is called the
    move has already been applied.
    """

    default_players: int = 2

    # Maximum number of half-moves (plies) before declaring a draw.
    MAX_HALF_MOVES: int = 200  # 100 full moves

    def __init__(self, config):
        super().__init__(config)
        self.board: chess.Board = chess.Board()
        self.move_history: list[str] = []  # SAN notation moves
        self.last_action: dict[str, Any] | None = None
        self._tools: dict[str, list[Callable]] = {}
        self._white_id: str = ""
        self._black_id: str = ""

    async def setup(self) -> None:
        """Initialize the chess board and create two LLM players."""
        if len(self.config.players) != 2:
            raise ValueError(
                f"Chess requires exactly 2 players, got {len(self.config.players)}"
            )

        # First player in config plays White, second plays Black
        white_info = self.config.players[0]
        black_info = self.config.players[1]

        self._white_id = white_info.player_id
        self._black_id = black_info.player_id

        # Create LLMPlayer instances with chess system instructions
        self.players[self._white_id] = LLMPlayer(
            info=white_info,
            client=self.client,
            system_instructions=CHESS_SYSTEM_PROMPT,
        )
        self.players[self._black_id] = LLMPlayer(
            info=black_info,
            client=self.client,
            system_instructions=CHESS_SYSTEM_PROMPT,
        )

        # Assign roles
        self.roles[self._white_id] = PlayerRole(
            name="White",
            team="white",
            description="Playing as White (moves first).",
            is_hidden=False,
        )
        self.roles[self._black_id] = PlayerRole(
            name="Black",
            team="black",
            description="Playing as Black.",
            is_hidden=False,
        )

        # Create tools (closures that capture this game instance)
        self._tools = create_chess_tools(self)

        # Initialize the board (fresh starting position)
        self.board = chess.Board()
        self.move_history = []
        self.last_action = None

        if self.config.verbose:
            print(f"Chess game {self.game_id} initialized.")
            print(f"  White: {white_info.name} ({white_info.model})")
            print(f"  Black: {black_info.name} ({black_info.model})")

    async def get_next_phase(self) -> GamePhase:
        """
        Return the next game phase.

        Chess alternates between White and Black action phases.
        The board.turn property tells us whose turn it is.
        """
        # Check if the game is already over
        if self.board.is_game_over() or len(self.move_history) >= self.MAX_HALF_MOVES:
            return GamePhase(
                phase_type=PhaseType.GAME_OVER,
                round_number=self.round_number,
                description="The game is over.",
                active_player_ids=[],
            )

        # Determine whose turn it is
        if self.board.turn == chess.WHITE:
            active_id = self._white_id
            color_name = "White"
        else:
            active_id = self._black_id
            color_name = "Black"

        # Increment round number on each White move (a full move)
        if self.board.turn == chess.WHITE:
            self.round_number = self.board.fullmove_number

        description = f"Move {self.board.fullmove_number}: {color_name} to play"
        if self.board.is_check():
            description += " (in check)"

        return GamePhase(
            phase_type=PhaseType.ACTION,
            round_number=self.round_number,
            description=description,
            active_player_ids=[active_id],
        )

    async def get_player_view(self, player_id: str) -> str:
        """Build the prompt string that shows this player the current board state."""
        if player_id == self._white_id:
            color = chess.WHITE
        else:
            color = chess.BLACK

        return get_turn_prompt(self.board, color, self.move_history)

    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        """Return the chess tool functions for the current action phase."""
        return self._tools.get("action", [])

    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        """
        Process the result of a player's action.

        Because the tools are closures that directly mutate the board, by the
        time this method is called the move has already been applied (if it was
        legal). We use self.last_action (set by the tool closure) to determine
        what happened.
        """
        # The last_action dict is set by the tool closures in tools.py
        last = self.last_action

        if last is None:
            # The LLM did not call any tool, or something unexpected happened.
            # Try to extract info from the action_result
            tool_name = "unknown"
            tools_called = getattr(action_result, "tools_called", [])
            if tools_called:
                tool_name = tools_called[-1]

            return ActionResult(
                player_id=player_id,
                action_name=tool_name,
                action_args={},
                result="No action was recorded. The LLM may not have called a tool correctly.",
                success=False,
                visible_to=None,
            )

        # Build the ActionResult from the stored last_action
        result = ActionResult(
            player_id=player_id,
            action_name=last.get("tool", "unknown"),
            action_args=last.get("args", {}),
            result=last.get("result", ""),
            success=last.get("success", False),
            visible_to=None,  # Chess is fully observable -- both players see everything
        )

        # Clear last_action for the next turn
        self.last_action = None

        return result

    async def check_game_over(self) -> GameOutcome | None:
        """
        Check whether the game has ended.

        Returns a GameOutcome if the game is over, None otherwise.
        Handles: checkmate, stalemate, insufficient material,
        fifty-move rule, threefold repetition, and max moves limit.
        """
        game_over = self.board.is_game_over()
        max_moves_reached = len(self.move_history) >= self.MAX_HALF_MOVES

        if not game_over and not max_moves_reached:
            return None

        metadata: dict[str, Any] = {
            "total_moves": len(self.move_history),
            "final_fen": self.board.fen(),
            "move_history": self.move_history.copy(),
        }

        # Determine the result
        if self.board.is_checkmate():
            # The side whose turn it is has been checkmated -- the OTHER side wins
            if self.board.turn == chess.WHITE:
                # White is checkmated, Black wins
                winner_ids = [self._black_id]
                loser_ids = [self._white_id]
                metadata["termination"] = "checkmate"
                metadata["winner_color"] = "black"
            else:
                # Black is checkmated, White wins
                winner_ids = [self._white_id]
                loser_ids = [self._black_id]
                metadata["termination"] = "checkmate"
                metadata["winner_color"] = "white"
        else:
            # All other endings are draws
            winner_ids = []
            loser_ids = []

            if self.board.is_stalemate():
                metadata["termination"] = "stalemate"
            elif self.board.is_insufficient_material():
                metadata["termination"] = "insufficient_material"
            elif self.board.is_fifty_moves():
                metadata["termination"] = "fifty_move_rule"
            elif self.board.is_repetition():
                metadata["termination"] = "threefold_repetition"
            elif max_moves_reached:
                metadata["termination"] = "max_moves_reached"
            else:
                metadata["termination"] = "unknown_draw"

        if self.config.verbose:
            termination = metadata["termination"]
            if winner_ids:
                winner_name = self.players[winner_ids[0]].info.name
                print(f"\nGame over: {termination}. Winner: {winner_name}")
            else:
                print(f"\nGame over: {termination}. Draw.")
            print(f"Total moves: {len(self.move_history)}")
            print(f"Final FEN: {self.board.fen()}")

        return GameOutcome(
            game_id=self.game_id,
            game_type="chess",
            winner_ids=winner_ids,
            loser_ids=loser_ids,
            ranking=winner_ids + loser_ids if winner_ids else None,
            metadata=metadata,
            timestamp=datetime.now(timezone.utc),
        )
