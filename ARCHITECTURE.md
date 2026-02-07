# LLMArena Architecture & Implementation Reference

## Overview

LLMArena is an LLM benchmarking arena where AI models compete against each other in 5 games: Chess, Poker, Mafia, Secret Hitler, and Impostor. Models interact agentically via tool calls through the Dedalus Labs SDK. Results are tracked with an ELO rating system.

**Stats**: 39 Python files, ~200KB of code, 5 fully implemented games.

---

## Technology Stack

- **Language**: Python 3.13
- **LLM Provider**: Dedalus Labs SDK (`dedalus_labs`) — unified, OpenAI-compatible API supporting Anthropic, OpenAI, Google, xAI, DeepSeek
- **Chess Engine**: `python-chess` for board state and move validation
- **CLI**: Click
- **Data Models**: Pydantic v2
- **Ratings Storage**: SQLite
- **Package Management**: pip with pyproject.toml (setuptools backend)

---

## Project Structure

```
LLMArena/
├── pyproject.toml                    # Package config, dependencies, CLI entry point
├── CLAUDE.md                         # Claude Code guidance
├── ARCHITECTURE.md                   # This file
├── .gitignore
├── llm_arena/
│   ├── __init__.py
│   ├── __main__.py                   # python -m llm_arena entry point
│   ├── cli.py                        # Click CLI: play, tournament, leaderboard, replay
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                  # Pydantic models (GameConfig, GamePhase, GameOutcome, etc.)
│   │   ├── game.py                   # BaseGame abstract class — the framework foundation
│   │   └── player.py                 # LLMPlayer wrapping Dedalus SDK
│   │
│   ├── games/
│   │   ├── __init__.py               # GAME_REGISTRY + @register_game decorator
│   │   ├── chess/
│   │   │   ├── __init__.py
│   │   │   ├── game.py               # ChessGame (268 lines)
│   │   │   ├── tools.py              # make_move, get_legal_moves
│   │   │   └── prompts.py            # Board rendering, UCI reference
│   │   ├── poker/
│   │   │   ├── __init__.py
│   │   │   ├── game.py               # PokerGame (655 lines)
│   │   │   ├── tools.py              # bet, call, raise_bet, fold, check
│   │   │   ├── hand_eval.py          # Full hand ranking engine
│   │   │   └── prompts.py            # Hole cards, community cards, pot info
│   │   ├── mafia/
│   │   │   ├── __init__.py
│   │   │   ├── game.py               # MafiaGame (551 lines)
│   │   │   ├── tools.py              # Discussion, voting, night action tools
│   │   │   ├── roles.py              # Mafia, Doctor, Detective, Villager
│   │   │   └── prompts.py            # Phase-specific prompts, role views
│   │   ├── secret_hitler/
│   │   │   ├── __init__.py
│   │   │   ├── game.py               # SecretHitlerGame (819 lines)
│   │   │   ├── tools.py              # Nomination, voting, policy, powers
│   │   │   ├── roles.py              # Liberal, Fascist, Hitler
│   │   │   └── prompts.py            # Per-phase prompts with role knowledge
│   │   └── impostor/
│   │       ├── __init__.py
│   │       ├── game.py               # ImpostorGame (685 lines)
│   │       ├── tools.py              # Movement, tasks, kills, reporting, voting
│   │       ├── roles.py              # Impostor, Crewmate + task generation
│   │       └── prompts.py            # Location-aware prompts
│   │
│   ├── ratings/
│   │   ├── __init__.py
│   │   ├── elo.py                    # ELO calculation (pairwise for multiplayer)
│   │   └── store.py                  # SQLite persistence
│   │
│   ├── logging/
│   │   ├── __init__.py
│   │   └── transcript.py             # JSON + human-readable game transcripts
│   │
│   ├── tournament/
│   │   ├── __init__.py
│   │   └── runner.py                 # Round-robin tournament runner
│   │
│   └── data/                         # Created at runtime
│       ├── ratings.db
│       └── logs/
```

---

## Core Framework Design

### Phase-Driven Game Loop (`core/game.py`)

The `BaseGame` abstract class defines a universal game loop that supports both simple turn-based games (Chess) and complex social deduction games (Mafia, Secret Hitler, Impostor).

**Main loop** (`BaseGame.run()`):
```
setup() → loop { get_next_phase() → run phase → check_game_over() }
```

**Two phase runners**:
- `_run_action_phase()` — Each active player takes a turn via tool calls. Players act on current state independently.
- `_run_discussion_phase()` — Players speak in sequence. Each player sees what previous players said (iterative discussion history).

**Six abstract methods** every game must implement:
1. `setup()` — Initialize game state, create LLMPlayer instances, assign roles
2. `get_next_phase()` → `GamePhase` — Define phase sequence
3. `get_player_view(player_id)` → `str` — Build filtered prompt (role-dependent visibility)
4. `get_tools_for_player(player_id)` → `list[Callable]` — Select tools for current phase/role
5. `process_action(player_id, action_result)` → `ActionResult` — Record what happened
6. `check_game_over()` → `GameOutcome | None` — Detect win/loss/draw

### Tool-Based LLM Interaction (`core/player.py`)

Each `LLMPlayer` wraps a `DedalusRunner` with:
- Isolated conversation history (`messages` list persisted via `result.to_input_list()`)
- `tool_choice="required"` to force structured actions (no free-text responses)
- Shared `AsyncDedalus` client for HTTP connection pooling

**Dedalus SDK integration**:
```python
result = await runner.run(
    input="game prompt",
    model="anthropic/claude-opus-4-6",
    instructions="system prompt",
    tools=[func1, func2],
    tool_choice="required",
    max_steps=3,
    temperature=0.7,
)
# result._RunResult:
#   .final_output (str) — last text response
#   .tool_results (list[dict]) — tool call results
#   .tools_called (list[str]) — function names called
#   .messages (list[dict]) — full conversation history
#   .to_input_list() — get messages for continuation
```

### Tool Closure Pattern

Every game defines tools as **closures over game state** in a factory function. This is the central design pattern — tools directly mutate game state and store their result for `process_action()` to read:

```python
def create_chess_tools(game: ChessGame) -> dict[str, list[Callable]]:
    def make_move(move_uci: str) -> str:
        """Make a chess move in UCI notation."""
        board = game.board
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            game.last_action = {"tool": "make_move", "success": False, "result": "Illegal move"}
            return "Illegal move"
        board.push(move)
        game.last_action = {"tool": "make_move", "success": True, "result": f"Played {move_uci}"}
        return f"Played {move_uci}"

    return {"action": [make_move]}
```

**Why closures?** The Dedalus SDK expects plain Python functions with type hints and docstrings. It auto-extracts JSON schemas from function signatures. Closures are the simplest way to bind game state while keeping the SDK happy.

### Game Registry (`games/__init__.py`)

```python
GAME_REGISTRY: dict[str, Type[BaseGame]] = {}

@register_game("chess")
class ChessGame(BaseGame): ...
```

Games self-register on import. The CLI imports all game modules to trigger registration.

---

## Game Implementations

### Chess (`games/chess/`)

**Complexity**: Simplest game. 2 players, perfect information, alternating turns.

| Aspect | Details |
|--------|---------|
| Players | 2 (White vs Black) |
| Phases | ACTION only (alternating White/Black) |
| Board | `chess.Board()` from python-chess |
| Tools | `make_move(move_uci)`, `get_legal_moves()` |
| Game Over | Checkmate, stalemate, insufficient material, 50-move rule, threefold repetition, max 200 half-moves |
| Validation | python-chess handles all legal move validation |

**Player view**: Unicode board (oriented from player's perspective), FEN string, material counts, full move history in standard notation.

### Poker (`games/poker/`)

**Complexity**: Hidden information (hole cards), multi-round betting, hand evaluation.

| Aspect | Details |
|--------|---------|
| Players | 2-6 (default 2, heads-up) |
| Starting Chips | 1000 per player |
| Blinds | 10/20 (configurable) |
| Phases | ACTION per player per betting action, RESOLUTION for hand start/end |
| Tools | `bet(amount)`, `call()`, `raise_bet(total_amount)`, `fold()`, `check()` |
| Game Over | One player remaining (all others at 0 chips), or max hands safety valve |

**Hand evaluation** (`hand_eval.py`):
- Evaluates all C(7,5)=21 combinations of 2 hole + 5 community cards
- Returns `(rank, tiebreaker_values)` for comparison
- Supports all 10 hand rankings: High Card through Royal Flush
- Handles wheel straights (A-2-3-4-5), split pots

**Betting logic**:
- Each "phase" is one player's turn to act
- After bet/raise, `_reopen_betting()` re-queues other players
- `_is_betting_complete()` checks all have acted and matched current bet
- Auto-folds players who fail to take valid action
- Handles all-in scenarios

**Hand lifecycle**: `_start_hand()` → deal hole cards → post blinds → pre-flop betting → flop → betting → turn → betting → river → betting → showdown → `_resolve_hand()`

### Mafia (`games/mafia/`)

**Complexity**: Social deduction with hidden roles, discussion, voting, and night actions.

| Aspect | Details |
|--------|---------|
| Players | 5-10 (default 7) |
| Roles | Mafia (2 for 7+, 1 for 5-6), Doctor (1), Detective (1), Villager (rest) |
| Phase Cycle | DISCUSSION → VOTING → DAY_RESOLUTION → NIGHT_ACTION → NIGHT_RESOLUTION → repeat |
| Win: Town | All Mafia eliminated |
| Win: Mafia | Mafia ≥ Town in alive count |

**Phase state machine** (`_PhaseStep`): Cyclic iterator over 5 steps. Round number increments at each new DISCUSSION.

**Tools by phase/role**:
- Discussion: `make_statement(statement)`, `accuse_player(player_name, reason)` — all players
- Voting: `cast_vote(player_name)` — all players
- Night (Mafia): `mafia_kill(player_name)` — first alive Mafia member only
- Night (Doctor): `doctor_protect(player_name)` — cannot protect same player consecutively
- Night (Detective): `detective_investigate(player_name)` — gets immediate Mafia/Not Mafia result

**Resolution logic**:
- Day: Tally votes, eliminate top-voted (ties = no elimination), reveal role
- Night: If kill target = protect target → saved. Otherwise target dies. Detective gets private result.

**Information hiding**: Mafia players see teammates' names. Detective gets private investigation results. Night kill visible only to Mafia team.

### Secret Hitler (`games/secret_hitler/`)

**Complexity**: Most complex game. Policy deck, elections, executive powers, term limits.

| Aspect | Details |
|--------|---------|
| Players | 5-10 (default 7) |
| Roles | Liberals (majority), Fascists (1-3), Hitler (1) |
| Policy Deck | 6 Liberal + 11 Fascist, shuffled |
| Win: Liberal | 5 Liberal policies OR Hitler executed |
| Win: Fascist | 6 Fascist policies OR Hitler elected Chancellor after 3+ Fascist policies |

**12 sub-phases**: discussion → nomination → voting → vote_result → president_discard → chancellor_enact → policy_enacted → power_* → round_end

**Key mechanics**:
- **Elections**: President nominates Chancellor → all vote Ja/Nein → majority passes
- **Policy enactment**: President draws 3, discards 1, passes 2 to Chancellor who enacts 1
- **Election tracker**: Failed votes increment counter. At 3 failures, top policy auto-enacted ("chaos")
- **Term limits**: Previous Chancellor always term-limited. Previous President term-limited with >5 alive players
- **Presidential powers** (after Fascist policies): Investigate loyalty, peek at deck, choose next president, execute a player

**Role knowledge**:
- Fascists see all fascist team members
- Hitler sees Fascists only in 5-6 player games
- Liberals see nothing

**Tools**: `make_statement`, `nominate_chancellor`, `cast_vote`, `discard_policy`, `enact_policy`, `investigate_player`, `execute_player`, `choose_next_president`

### Impostor (`games/impostor/`)

**Complexity**: Location-based social deduction with tasks, kills, and meetings.

| Aspect | Details |
|--------|---------|
| Players | 4-10 (default 6) |
| Roles | Impostors (1-2), Crewmates (rest) |
| Locations | Cafeteria, Reactor, Electrical, MedBay, Navigation, Security, Admin, Storage |
| Win: Crew | All tasks completed OR all impostors ejected |
| Win: Impostor | Living impostors ≥ living crewmates |

**Phase flow**: ACTION rounds (free roam) → if body reported or emergency meeting → DISCUSSION → VOTING → RESOLUTION → back to ACTION

**Key mechanics**:
- Players can only see others at their same location
- Each crewmate has 3 assigned tasks at specific locations
- Impostors can fake tasks and kill crewmates (2-round cooldown)
- Dead players leave bodies that can be reported
- Limited emergency meetings per player
- Task progress tracked as percentage

**Tools by role**:
- Crew: `move_to(location)`, `do_task()`, `report_body()`, `call_emergency_meeting()`
- Impostor: all crew tools + `kill_player(player_name)`
- Discussion: `make_statement(statement)`
- Voting: `cast_vote(player_name)`, `skip_vote()`

**Custom game loop**: Overrides `BaseGame.run()` to dispatch all 4 phase types (ACTION, DISCUSSION, VOTING, RESOLUTION) since the base class only natively handles ACTION and DISCUSSION.

---

## Supporting Systems

### ELO Rating System (`ratings/`)

**Algorithm**: Standard ELO with pairwise extension for multiplayer games.

```
Expected score: E(A) = 1 / (1 + 10^((Rb - Ra) / 400))
Rating update: R'(A) = R(A) + K * (S - E) / (N-1)
```

- K-factor: 32
- Default rating: 1500
- Multiplayer: Each player compared pairwise against every other, deltas averaged

**Storage** (`store.py`): SQLite with two tables:
- `ratings` — model, rating, games_played, wins, losses, last_updated
- `game_results` — game_id, game_type, players, winners, losers, metadata, timestamp

### Transcript Logger (`logging/transcript.py`)

Outputs two files per game in `data/logs/`:
- `{game_id}.json` — Structured event log (game_start, phase_change, action, game_end)
- `{game_id}.txt` — Human-readable transcript

### Tournament Runner (`tournament/runner.py`)

`TournamentRunner.run_round_robin()`:
- Generates all C(n, k) player combinations
- Runs `games_per_matchup` games for each combination
- Updates ELO after each game

### CLI (`cli.py`)

```bash
llm-arena play <game> -m <model1> -m <model2> [-v]
llm-arena tournament <game> -m <model1> -m <model2> -m <model3> [-n games]
llm-arena leaderboard
llm-arena replay <game_id>
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Phase-driven loop** (not event-driven) | Simpler to reason about and debug for a hackathon. Each phase has clear inputs/outputs. |
| **Closures for tools** (not classes) | Dedalus SDK expects plain functions. Closures bind game state with minimal boilerplate. |
| **`tool_choice="required"`** | Forces LLMs to call tools rather than respond with free text. Guarantees structured game actions. |
| **Per-player `DedalusRunner`** | Isolates conversation histories. Shared `AsyncDedalus` client for HTTP efficiency. |
| **SQLite for ratings** | Single-file, handles concurrent writes, supports queries. No server needed. |
| **Pairwise ELO for multiplayer** | Standard extension (used by TrueSkill). Works for both 2-player and N-player games. |
| **`last_action` bridge pattern** | Tools execute immediately (mutating state) but `process_action()` needs to know what happened. The tool stores its result in game state for later reading. |
| **Separate prompts.py per game** | Keeps game logic and LLM prompting cleanly separated. Prompts can be tuned independently. |

---

## Setup & Installation

```bash
# Requires Python 3.11+ (tested on 3.13)
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

# Set API key
export DEDALUS_API_KEY=your-key-here
```

**Dependencies** (from pyproject.toml):
- `dedalus_labs` — LLM API
- `click` — CLI framework
- `pydantic>=2.0` — Data models
- `python-chess` — Chess game engine

---

## Adding a New Game

1. Create `games/<name>/` with `__init__.py`, `game.py`, `tools.py`, `prompts.py` (and `roles.py` if social deduction)
2. In `game.py`: subclass `BaseGame`, decorate with `@register_game("name")`, set `default_players`, implement 6 abstract methods
3. In `tools.py`: create factory function returning closures that mutate game state and set `game.state["last_action"]`
4. In `prompts.py`: write system prompt and per-phase prompt builders with role-appropriate information filtering
5. In `__init__.py`: `from llm_arena.games.<name>.game import <GameClass>`
6. In `cli.py`: add import `import llm_arena.games.<name>` and add to `GAME_CHOICES`

---

## Testing

All tests were run during development:

- **Core imports**: All 39 Python files parse, all modules import cleanly
- **Game registration**: All 5 games appear in `GAME_REGISTRY`
- **ELO system**: Pairwise calculation tested for 2-player and 4-player scenarios
- **Rating store**: SQLite create/read/update verified with temp databases
- **Chess**: Setup, make_move (legal and illegal), get_legal_moves, checkmate detection (fool's mate)
- **Poker hand eval**: All 10 hand rankings tested, tiebreaker comparison, split pots
- **Poker setup**: Player creation, chip allocation, tool creation
- **Mafia roles**: Correct distribution for 5 and 7 players
- **Mafia setup**: Player creation, role assignment, alive tracking
- **Secret Hitler roles**: Correct Liberal/Fascist/Hitler distribution for 7 players
- **Secret Hitler setup**: Policy deck (17 cards), player creation, state initialization
- **Impostor roles**: Correct impostor/crewmate distribution for 6 players
- **Impostor tasks**: 3 tasks per crewmate, 0 for impostors
- **Impostor setup**: Player creation, task total, alive tracking
- **CLI**: Leaderboard (empty state), help text, play command help, game choices
