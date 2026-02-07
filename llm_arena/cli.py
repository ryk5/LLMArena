from __future__ import annotations

import asyncio
from pathlib import Path

import click

from llm_arena.core.types import GameConfig, PlayerInfo

# Import game modules to trigger registration
import llm_arena.games.chess  # noqa: F401
import llm_arena.games.poker  # noqa: F401
import llm_arena.games.mafia  # noqa: F401
import llm_arena.games.secret_hitler  # noqa: F401
import llm_arena.games.impostor  # noqa: F401

from llm_arena.games import GAME_REGISTRY
from llm_arena.ratings.elo import EloRating
from llm_arena.ratings.store import RatingStore
from llm_arena.tournament.runner import TournamentRunner

GAME_CHOICES = ["chess", "poker", "mafia", "secret_hitler", "impostor"]


@click.group()
def cli():
    """LLMArena - Pit LLMs against each other in games."""
    pass


@cli.command()
@click.argument("game_type", type=click.Choice(GAME_CHOICES))
@click.option("--models", "-m", multiple=True, required=True, help="Model ID (e.g. anthropic/claude-opus-4-6)")
@click.option("--verbose", "-v", is_flag=True)
def play(game_type: str, models: tuple[str, ...], verbose: bool):
    """Run a single game."""
    if game_type not in GAME_REGISTRY:
        click.echo(f"Game '{game_type}' not yet implemented.")
        return

    players = []
    seen: dict[str, int] = {}
    for m in models:
        seen[m] = seen.get(m, 0) + 1
        suffix = f"-{seen[m]}" if seen[m] > 1 or models.count(m) > 1 else ""
        short = m.split("/")[-1]
        players.append(
            PlayerInfo(player_id=f"{m}{suffix}", name=f"{short}{suffix}", model=m)
        )
    config = GameConfig(game_type=game_type, players=players, verbose=verbose)
    game_cls = GAME_REGISTRY[game_type]
    game = game_cls(config)

    click.echo(f"\nStarting {game_type} game...")
    click.echo(f"Players: {', '.join(m.split('/')[-1] for m in models)}\n")

    outcome = asyncio.run(game.run())

    click.echo(f"\nGame Over!")
    click.echo(f"  Winners: {[m.split('/')[-1] for m in outcome.winner_ids]}")
    click.echo(f"  Losers:  {[m.split('/')[-1] for m in outcome.loser_ids]}")

    # Update ELO — map player_ids back to model names
    id_to_model = {p.player_id: p.model for p in players}
    outcome.winner_ids = [id_to_model.get(pid, pid) for pid in outcome.winner_ids]
    outcome.loser_ids = [id_to_model.get(pid, pid) for pid in outcome.loser_ids]
    if outcome.ranking:
        outcome.ranking = [id_to_model.get(pid, pid) for pid in outcome.ranking]
    store = RatingStore()
    current_ratings = {m: store.get_rating(m) for m in models}
    new_ratings = EloRating.update_ratings(current_ratings, outcome)
    store.update_ratings(new_ratings, outcome)
    store.close()

    click.echo(f"\nELO updated. Run 'llm-arena leaderboard' to see standings.")


@cli.command()
@click.argument("game_type", type=click.Choice(GAME_CHOICES))
@click.option("--models", "-m", multiple=True, required=True)
@click.option("--games", "-n", default=3, help="Games per matchup")
def tournament(game_type: str, models: tuple[str, ...], games: int):
    """Run a round-robin tournament."""
    if game_type not in GAME_REGISTRY:
        click.echo(f"Game '{game_type}' not yet implemented.")
        return

    runner = TournamentRunner()
    outcomes = asyncio.run(
        runner.run_round_robin(game_type, list(models), games_per_matchup=games)
    )

    click.echo(f"\nTournament complete. {len(outcomes)} games played.")
    _print_leaderboard()


@cli.command()
def leaderboard():
    """Show current ELO ratings."""
    _print_leaderboard()


@cli.command()
@click.argument("game_id")
def replay(game_id: str):
    """Print the transcript of a past game."""
    path = Path(f"data/logs/{game_id}.txt")
    if path.exists():
        click.echo(path.read_text())
    else:
        click.echo(f"No transcript found for game '{game_id}'")


def _print_leaderboard():
    store = RatingStore()
    ratings = store.get_all_ratings()
    store.close()

    if not ratings:
        click.echo("\nNo ratings yet. Play some games first!")
        return

    click.echo(f"\n{'Model':<35} {'Rating':>7} {'W/L':>7} {'Games':>6}")
    click.echo("-" * 58)
    for r in ratings:
        click.echo(
            f"{r['model']:<35} {r['rating']:7.0f} {r['wins']}/{r['losses']:<4} {r['games']:>5}"
        )
