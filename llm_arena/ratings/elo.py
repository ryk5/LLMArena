from __future__ import annotations

from llm_arena.core.types import GameOutcome


class EloRating:
    """Standard ELO rating calculator with pairwise extension for multiplayer."""

    DEFAULT_RATING = 1500.0
    K_FACTOR = 32

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def update_ratings(
        ratings: dict[str, float],
        outcome: GameOutcome,
    ) -> dict[str, float]:
        """
        Update ratings for all players based on game outcome.
        Uses pairwise comparison for multiplayer games.
        """
        new_ratings = dict(ratings)
        all_players = outcome.winner_ids + outcome.loser_ids

        if len(all_players) < 2:
            return new_ratings

        for i, p1 in enumerate(all_players):
            total_delta = 0.0
            for j, p2 in enumerate(all_players):
                if i == j:
                    continue
                expected = EloRating.expected_score(ratings[p1], ratings[p2])
                if p1 in outcome.winner_ids and p2 in outcome.loser_ids:
                    actual = 1.0
                elif p1 in outcome.loser_ids and p2 in outcome.winner_ids:
                    actual = 0.0
                else:
                    actual = 0.5
                total_delta += EloRating.K_FACTOR * (actual - expected)

            new_ratings[p1] = ratings[p1] + total_delta / (len(all_players) - 1)

        return new_ratings
