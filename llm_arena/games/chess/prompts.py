from __future__ import annotations

import chess


CHESS_SYSTEM_PROMPT = """\
You are a chess-playing AI competing in a tournament against another AI model.

Rules:
- You will be told which color you are playing (White or Black).
- On your turn you will see the current board state and move history.
- You MUST call the `make_move` tool with a legal move in UCI notation (e.g. "e2e4", "g1f3", "e7e8q" for promotion).
- Legal moves are listed in the prompt — pick one and call `make_move`.
- Think carefully about strategy: control the center, develop pieces, protect your king, and look for tactical opportunities.
- You may only make ONE move per turn.
- Be concise. Do not write long explanations — just pick your move and call the tool.

UCI notation quick reference:
- A move is the starting square followed by the destination square: "e2e4"
- Castling kingside: "e1g1" (White) or "e8g8" (Black)
- Castling queenside: "e1c1" (White) or "e8c8" (Black)
- Pawn promotion appends the piece letter: "e7e8q" (promote to queen)

Play your best chess. Good luck!
"""


def get_turn_prompt(board: chess.Board, color: chess.Color, move_history: list[str]) -> str:
    """Build the per-turn prompt that shows the player the current game state."""
    color_name = "White" if color == chess.WHITE else "Black"
    opponent_color = "Black" if color == chess.WHITE else "White"

    # Determine the move number (full moves, 1-indexed)
    fullmove = board.fullmove_number

    # Build move history display
    if move_history:
        history_lines = _format_move_history(move_history)
    else:
        history_lines = "(No moves yet -- you are making the first move.)"

    # Check / checkmate / stalemate status
    status_line = ""
    if board.is_check():
        status_line = f"\n** {color_name} is in CHECK! You must resolve the check. **"

    # Count material
    material = _count_material(board)

    prompt = f"""\
## Your Turn -- {color_name} to move (move {fullmove})

### Board
```
{board.unicode(borders=True, orientation=color)}
```

FEN: `{board.fen()}`
{status_line}

### Material
White: {material['white']}
Black: {material['black']}

### Move History
{history_lines}

### Legal Moves
{_format_legal_moves(board)}

### Instructions
It is {color_name}'s turn. You are playing {color_name} against {opponent_color}.
Analyze the position, then call `make_move` with your chosen move in UCI notation.
"""
    return prompt


def _format_legal_moves(board: chess.Board) -> str:
    """Format legal moves grouped by piece for the prompt."""
    if board.is_game_over():
        return "No legal moves — game is over."
    moves_by_piece: dict[str, list[str]] = {}
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece is not None:
            piece_name = chess.piece_name(piece.piece_type).capitalize()
        else:
            piece_name = "Unknown"
        from_sq = chess.square_name(move.from_square)
        label = f"{piece_name} on {from_sq}"
        if label not in moves_by_piece:
            moves_by_piece[label] = []
        moves_by_piece[label].append(move.uci())
    lines: list[str] = []
    for label, uci_moves in sorted(moves_by_piece.items()):
        lines.append(f"  {label}: {', '.join(uci_moves)}")
    return "\n".join(lines)


def _format_move_history(move_history: list[str]) -> str:
    """Format the move history as numbered move pairs."""
    lines: list[str] = []
    for i in range(0, len(move_history), 2):
        move_num = i // 2 + 1
        white_move = move_history[i]
        if i + 1 < len(move_history):
            black_move = move_history[i + 1]
            lines.append(f"{move_num}. {white_move} {black_move}")
        else:
            lines.append(f"{move_num}. {white_move}")
    return "\n".join(lines)


def _count_material(board: chess.Board) -> dict[str, str]:
    """Return a human-readable summary of each side's material."""
    piece_values = {
        chess.PAWN: ("P", 1),
        chess.KNIGHT: ("N", 3),
        chess.BISHOP: ("B", 3),
        chess.ROOK: ("R", 5),
        chess.QUEEN: ("Q", 9),
        chess.KING: ("K", 0),
    }

    result: dict[str, str] = {}
    for color, color_name in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
        pieces: list[str] = []
        total = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            count = len(board.pieces(piece_type, color))
            if count > 0:
                symbol, value = piece_values[piece_type]
                pieces.append(f"{symbol}x{count}")
                total += count * value
        result[color_name] = f"{', '.join(pieces)} (value: {total})"

    return result
