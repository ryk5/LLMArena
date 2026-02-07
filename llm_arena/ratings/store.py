from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from llm_arena.core.types import GameOutcome
from llm_arena.ratings.elo import EloRating


class RatingStore:
    """SQLite-backed persistent storage for ELO ratings and game history."""

    def __init__(self, db_path: str = "data/ratings.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS ratings (
                model TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0,
                games_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                last_updated TEXT
            );
            CREATE TABLE IF NOT EXISTS game_results (
                game_id TEXT PRIMARY KEY,
                game_type TEXT,
                players TEXT,
                winners TEXT,
                losers TEXT,
                metadata TEXT,
                timestamp TEXT
            );
        """)
        self.conn.commit()

    def get_rating(self, model: str) -> float:
        row = self.conn.execute(
            "SELECT rating FROM ratings WHERE model = ?", (model,)
        ).fetchone()
        if row is None:
            self._create_player(model)
            return EloRating.DEFAULT_RATING
        return row[0]

    def get_all_ratings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT model, rating, games_played, wins, losses FROM ratings ORDER BY rating DESC"
        ).fetchall()
        return [
            {"model": r[0], "rating": r[1], "games": r[2], "wins": r[3], "losses": r[4]}
            for r in rows
        ]

    def update_ratings(self, new_ratings: dict[str, float], outcome: GameOutcome):
        now = datetime.now(timezone.utc).isoformat()
        for model, rating in new_ratings.items():
            # Ensure player exists
            self._create_player(model)
            is_winner = model in outcome.winner_ids
            self.conn.execute(
                """
                UPDATE ratings SET
                    rating = ?,
                    games_played = games_played + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    last_updated = ?
                WHERE model = ?
                """,
                (rating, int(is_winner), int(not is_winner), now, model),
            )

        self.conn.execute(
            """
            INSERT INTO game_results (game_id, game_type, players, winners, losers, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.game_id,
                outcome.game_type,
                json.dumps(outcome.winner_ids + outcome.loser_ids),
                json.dumps(outcome.winner_ids),
                json.dumps(outcome.loser_ids),
                json.dumps(outcome.metadata),
                now,
            ),
        )
        self.conn.commit()

    def _create_player(self, model: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO ratings (model, last_updated) VALUES (?, ?)",
            (model, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
