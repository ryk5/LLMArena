from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import chess

if TYPE_CHECKING:
    from llm_arena.games.chess.game import ChessGame


def create_chess_tools(game: ChessGame) -> dict[str, list[Callable]]:
    """
    Factory that creates tool functions for the chess game.

    The tools are closures that capture the game instance so they can
    directly read and mutate the board state. This is the pattern expected
    by the Dedalus SDK: plain functions with type hints and docstrings.

    Returns a dict keyed by phase name -> list of tool callables.
    Chess only has "action" phase tools.
    """

    def make_move(move_uci: str) -> str:
        """Make a chess move in UCI notation (e.g. 'e2e4', 'g1f3', 'e7e8q').

        Args:
            move_uci: The move in UCI notation -- source square followed by
                      destination square, with optional promotion piece.
        """
        board: chess.Board = game.board

        # Parse the UCI string
        try:
            move = chess.Move.from_uci(move_uci.strip().lower())
        except (ValueError, chess.InvalidMoveError):
            game.last_action = {
                "tool": "make_move",
                "args": {"move_uci": move_uci},
                "success": False,
                "result": f"Invalid UCI format: '{move_uci}'. Use format like 'e2e4'.",
            }
            return game.last_action["result"]

        # Check legality
        if move not in board.legal_moves:
            legal = ", ".join(m.uci() for m in board.legal_moves)
            game.last_action = {
                "tool": "make_move",
                "args": {"move_uci": move_uci},
                "success": False,
                "result": (
                    f"Illegal move: '{move_uci}'. "
                    f"Legal moves are: {legal}"
                ),
            }
            return game.last_action["result"]

        # Generate SAN notation before pushing (needs current board state)
        san = board.san(move)

        # Apply the move
        board.push(move)

        # Record in game's move history
        game.move_history.append(san)

        # Build result string
        result_parts = [f"Move played: {san} ({move.uci()})"]

        if board.is_checkmate():
            result_parts.append("CHECKMATE!")
        elif board.is_check():
            result_parts.append("Check!")
        elif board.is_stalemate():
            result_parts.append("Stalemate -- draw.")
        elif board.is_insufficient_material():
            result_parts.append("Draw by insufficient material.")
        elif board.is_fifty_moves():
            result_parts.append("Draw by fifty-move rule.")
        elif board.is_repetition():
            result_parts.append("Draw by threefold repetition.")

        result_str = " ".join(result_parts)

        game.last_action = {
            "tool": "make_move",
            "args": {"move_uci": move.uci()},
            "success": True,
            "result": result_str,
            "san": san,
        }

        return result_str

    return {
        "action": [make_move],
    }
