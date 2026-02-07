from __future__ import annotations

import itertools

from llm_arena.core.types import GameConfig, GameOutcome, PlayerInfo
from llm_arena.games import GAME_REGISTRY
from llm_arena.ratings.elo import EloRating
from llm_arena.ratings.store import RatingStore


class TournamentRunner:
    """Run multiple games across model combinations and track ratings."""

    def __init__(self, store: RatingStore | None = None):
        self.store = store or RatingStore()

    async def run_round_robin(
        self,
        game_type: str,
        models: list[str],
        games_per_matchup: int = 3,
        players_per_game: int | None = None,
    ) -> list[GameOutcome]:
        game_cls = GAME_REGISTRY[game_type]
        n_players = players_per_game or game_cls.default_player_count()

        outcomes: list[GameOutcome] = []
        matchups = list(itertools.combinations(models, n_players))

        for matchup in matchups:
            print(f"\nMatchup: {' vs '.join(matchup)}")
            for game_num in range(games_per_matchup):
                players = [
                    PlayerInfo(
                        player_id=model,
                        name=model.split("/")[-1],
                        model=model,
                    )
                    for model in matchup
                ]
                config = GameConfig(game_type=game_type, players=players, verbose=True)
                game = game_cls(config)
                outcome = await game.run()
                outcomes.append(outcome)

                current_ratings = {m: self.store.get_rating(m) for m in matchup}
                new_ratings = EloRating.update_ratings(current_ratings, outcome)
                self.store.update_ratings(new_ratings, outcome)

                winner_names = [m.split("/")[-1] for m in outcome.winner_ids]
                print(f"  Game {game_num + 1}: Winners = {winner_names}")

        return outcomes
