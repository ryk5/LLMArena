from __future__ import annotations

from typing import Type

from llm_arena.core.game import BaseGame

GAME_REGISTRY: dict[str, Type[BaseGame]] = {}


def register_game(name: str):
    """Decorator to register a game class."""

    def decorator(cls: Type[BaseGame]):
        GAME_REGISTRY[name] = cls
        return cls

    return decorator
