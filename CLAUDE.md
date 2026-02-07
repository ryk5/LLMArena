# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLMArena is a benchmarking arena where AI models compete against each other in games (Chess, Poker, Mafia, Secret Hitler, Impostor). Models interact agentically via tool calls through the Dedalus Labs SDK.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires `DEDALUS_API_KEY` env var for LLM API access.

## CLI Commands

```bash
llm-arena play chess -m anthropic/claude-opus-4-6 -m openai/gpt-5.2 -v
llm-arena play mafia -m model1 -m model2 -m model3 -m model4 -m model5 -m model6 -m model7
llm-arena tournament chess -m model1 -m model2 -m model3 -n 5
llm-arena leaderboard
llm-arena replay <game_id>
```

## Architecture

### Phase-Driven Game Loop

All games inherit from `BaseGame` (`core/game.py`). The main loop cycles through phases:
- `_run_discussion_phase()` — iterative: each player speaks, seeing prior statements
- `_run_action_phase()` — each active player calls tools on current state

Each game subclass defines its own phase sequence via `get_next_phase()`. Chess uses only ACTION phases. Social deduction games (Mafia, Secret Hitler, Impostor) cycle through DISCUSSION -> VOTING -> ACTION -> RESOLUTION.

### Tool-Based LLM Interaction

Games expose actions as **plain Python functions with type hints + docstrings** (the Dedalus SDK auto-extracts schemas). Tool functions are closures over game state, created by factory functions in each game's `tools.py`:

```python
def create_chess_tools(game: ChessGame) -> dict[str, list[Callable]]:
    def make_move(move_uci: str) -> str:
        """Make a chess move..."""
        # directly mutates game.board
```

Tools store their result in `game.state["last_action"]` (or `game.last_action` for Chess) so `process_action()` can build an `ActionResult`.

### Per-Player Isolation

Each `LLMPlayer` (`core/player.py`) wraps a `DedalusRunner` with isolated conversation history. All share one `AsyncDedalus` client. Tool calls use `tool_choice="required"` to force structured actions.

### Key Patterns

- **Game registry**: `@register_game("name")` decorator in `games/__init__.py`
- **Player views**: `get_player_view(player_id)` returns a filtered prompt — hidden info is role-dependent
- **ELO ratings**: Pairwise extension for multiplayer games, SQLite-backed (`data/ratings.db`)
- **Transcripts**: JSON + human-readable `.txt` logs in `data/logs/`

## Module Layout

- `core/types.py` — Pydantic models: `GameConfig`, `GamePhase`, `GameOutcome`, `ActionResult`, `PlayerRole`, `PlayerInfo`
- `core/game.py` — `BaseGame` abstract class (the framework's foundation)
- `core/player.py` — `LLMPlayer` wrapping Dedalus SDK
- `games/<name>/game.py` — Game subclass with `@register_game`
- `games/<name>/tools.py` — Tool factory returning closures over game state
- `games/<name>/prompts.py` — System prompts and per-turn prompt builders
- `games/<name>/roles.py` — Role definitions and assignment (social deduction games)
- `ratings/elo.py` — ELO calculation, `ratings/store.py` — SQLite persistence
- `tournament/runner.py` — Round-robin tournament runner
- `cli.py` — Click CLI entry point

## Adding a New Game

1. Create `games/<name>/` with `game.py`, `tools.py`, `prompts.py`, `__init__.py`
2. Subclass `BaseGame`, decorate with `@register_game("name")`
3. Implement the 6 abstract methods: `setup`, `get_next_phase`, `get_player_view`, `get_tools_for_player`, `process_action`, `check_game_over`
4. Create tool factory: `create_<name>_tools(game)` returning closures
5. Import the game module in `__init__.py` and add the import in `cli.py`

## Dedalus SDK Usage

```python
from dedalus_labs import AsyncDedalus
from dedalus_labs.lib.runner import DedalusRunner

client = AsyncDedalus()  # uses DEDALUS_API_KEY env var
runner = DedalusRunner(client)
result = await runner.run(
    input="prompt",
    model="anthropic/claude-opus-4-6",  # provider/model format
    tools=[func1, func2],               # plain Python functions
    tool_choice="required",
    instructions="system prompt",
)
# result.final_output, result.tool_results, result.tools_called, result.messages
```
