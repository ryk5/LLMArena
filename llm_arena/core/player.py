from __future__ import annotations

import asyncio
from typing import Any, Callable

from dedalus_labs import AsyncDedalus
from dedalus_labs.lib.runner import DedalusRunner
from pydantic import BaseModel

from llm_arena.core.types import GamePhase, PlayerInfo


class LLMPlayer:
    """
    Wraps a single LLM identity in a game.
    Each player gets its own DedalusRunner with separate conversation history.
    """

    def __init__(
        self,
        info: PlayerInfo,
        client: AsyncDedalus,
        system_instructions: str = "",
    ):
        self.info = info
        self.client = client
        self.runner = DedalusRunner(client)
        self.system_instructions = system_instructions
        self.messages: list[dict] = []

    async def take_action(
        self,
        game_prompt: str,
        tools: list[Callable],
        phase: GamePhase,
        response_format: type[BaseModel] | None = None,
        max_steps: int = 1,
    ) -> Any:
        """
        Ask the LLM to take an action given the current game state.
        Returns the RunResult from Dedalus.
        """
        kwargs: dict[str, Any] = dict(
            input=game_prompt,
            model=self.info.model,
            instructions=self.system_instructions,
            tools=tools,
            tool_choice={"type": "any"},
            max_steps=max_steps,
            temperature=0.7,
            max_tokens=512,
        )

        if response_format:
            kwargs["response_format"] = response_format

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                result = await self.runner.run(**kwargs)
                return result
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
        raise last_err

    def reset_history(self):
        """Clear conversation history between games."""
        self.messages = []
