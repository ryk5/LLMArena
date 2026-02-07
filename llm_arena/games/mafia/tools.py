from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_arena.games.mafia.game import MafiaGame


def _get_alive_names(game: MafiaGame) -> list[str]:
    """Return display names of all alive players."""
    return [
        game.players[pid].info.name
        for pid in game.state["alive"]
    ]


def _name_to_id(game: MafiaGame, name: str) -> str | None:
    """Resolve a display name to a player ID. Returns None if not found."""
    for pid, player in game.players.items():
        if player.info.name == name:
            return pid
    return None


def _is_alive(game: MafiaGame, player_id: str) -> bool:
    """Check whether a player is currently alive."""
    return player_id in game.state["alive"]


def _validate_target(
    game: MafiaGame,
    player_name: str,
    *,
    exclude_self_id: str | None = None,
    exclude_team: str | None = None,
) -> tuple[str | None, str | None]:
    """
    Validate a target by display name.

    Returns (player_id, error_message). If error_message is not None, the
    target is invalid.
    """
    target_id = _name_to_id(game, player_name)
    if target_id is None:
        alive_names = _get_alive_names(game)
        return None, (
            f"No player named '{player_name}'. "
            f"Living players: {', '.join(alive_names)}"
        )

    if not _is_alive(game, target_id):
        return None, f"{player_name} is already eliminated."

    if exclude_self_id and target_id == exclude_self_id:
        return None, "You cannot target yourself."

    if exclude_team:
        role = game.roles[target_id]
        if role.team == exclude_team:
            return None, f"{player_name} is on your team. Choose someone else."

    return target_id, None


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def create_mafia_tools(game: MafiaGame) -> dict[str, Callable]:
    """
    Create all tool functions for the Mafia game.

    Returns a dict keyed by tool category:
        "discussion" -> [make_statement, accuse_player]
        "voting"     -> [cast_vote]
        "mafia"      -> [mafia_kill]
        "doctor"     -> [doctor_protect]
        "detective"  -> [detective_investigate]
    """

    # -----------------------------------------------------------------------
    # Discussion tools
    # -----------------------------------------------------------------------

    def make_statement(statement: str) -> str:
        """Make a public statement during the day discussion.

        Args:
            statement: What you want to say to the group. Be strategic --
                       share observations, defend yourself, or cast suspicion.

        Returns:
            Confirmation of your statement.
        """
        game.state["last_action"] = {
            "tool": "make_statement",
            "args": {"statement": statement},
        }
        return statement

    def accuse_player(player_name: str, reason: str) -> str:
        """Publicly accuse a player of being Mafia.

        Args:
            player_name: The display name of the player you are accusing.
            reason: Your reasoning for the accusation.

        Returns:
            A formatted accusation string, or an error if the target is invalid.
        """
        target_id, error = _validate_target(game, player_name)
        if error:
            game.state["last_action"] = {
                "tool": "accuse_player",
                "args": {"player_name": player_name, "reason": reason},
                "error": error,
            }
            return f"Invalid accusation: {error}"

        accusation = f"I accuse {player_name}! {reason}"
        game.state["last_action"] = {
            "tool": "accuse_player",
            "args": {"player_name": player_name, "reason": reason},
        }
        return accusation

    # -----------------------------------------------------------------------
    # Voting tools
    # -----------------------------------------------------------------------

    def cast_vote(player_name: str) -> str:
        """Vote to eliminate a player during the day vote.

        Args:
            player_name: The display name of the player you want to eliminate.

        Returns:
            Confirmation of your vote, or an error if the target is invalid.
        """
        target_id, error = _validate_target(game, player_name)
        if error:
            game.state["last_action"] = {
                "tool": "cast_vote",
                "args": {"player_name": player_name},
                "error": error,
            }
            return f"Invalid vote: {error}"

        game.state["votes"][target_id] = game.state["votes"].get(target_id, 0) + 1
        game.state["last_action"] = {
            "tool": "cast_vote",
            "args": {"player_name": player_name},
            "target_id": target_id,
        }
        return f"You voted to eliminate {player_name}."

    # -----------------------------------------------------------------------
    # Night tools -- Mafia
    # -----------------------------------------------------------------------

    def mafia_kill(player_name: str) -> str:
        """Choose a player to kill tonight (Mafia only).

        Args:
            player_name: The display name of the player the Mafia wants to eliminate.

        Returns:
            Confirmation of the kill target, or an error if the target is invalid.
        """
        target_id, error = _validate_target(
            game, player_name, exclude_team="mafia",
        )
        if error:
            game.state["last_action"] = {
                "tool": "mafia_kill",
                "args": {"player_name": player_name},
                "error": error,
            }
            return f"Invalid target: {error}"

        game.state["night_actions"]["mafia_kill"] = target_id
        game.state["last_action"] = {
            "tool": "mafia_kill",
            "args": {"player_name": player_name},
            "target_id": target_id,
        }
        return f"The Mafia will target {player_name} tonight."

    # -----------------------------------------------------------------------
    # Night tools -- Doctor
    # -----------------------------------------------------------------------

    def doctor_protect(player_name: str) -> str:
        """Choose a player to protect tonight (Doctor only).

        Args:
            player_name: The display name of the player you want to protect
                         from being killed tonight.

        Returns:
            Confirmation of the protection, or an error if the target is invalid.
        """
        target_id, error = _validate_target(game, player_name)
        if error:
            game.state["last_action"] = {
                "tool": "doctor_protect",
                "args": {"player_name": player_name},
                "error": error,
            }
            return f"Invalid target: {error}"

        # Check consecutive protection restriction
        last_protected = game.state.get("last_doctor_protect")
        if last_protected and last_protected == target_id:
            protected_name = game.players[target_id].info.name
            game.state["last_action"] = {
                "tool": "doctor_protect",
                "args": {"player_name": player_name},
                "error": "same_target",
            }
            return (
                f"You cannot protect {protected_name} two nights in a row. "
                "Choose someone else."
            )

        game.state["night_actions"]["doctor_protect"] = target_id
        game.state["last_action"] = {
            "tool": "doctor_protect",
            "args": {"player_name": player_name},
            "target_id": target_id,
        }
        return f"You will protect {player_name} tonight."

    # -----------------------------------------------------------------------
    # Night tools -- Detective
    # -----------------------------------------------------------------------

    def detective_investigate(player_name: str) -> str:
        """Investigate a player to learn if they are Mafia (Detective only).

        Args:
            player_name: The display name of the player you want to investigate.

        Returns:
            The investigation result revealing whether the player is Mafia,
            or an error if the target is invalid.
        """
        target_id, error = _validate_target(game, player_name)
        if error:
            game.state["last_action"] = {
                "tool": "detective_investigate",
                "args": {"player_name": player_name},
                "error": error,
            }
            return f"Invalid target: {error}"

        game.state["night_actions"]["detective_investigate"] = target_id

        # Immediately reveal the result to the Detective
        target_role = game.roles[target_id]
        is_mafia = target_role.team == "mafia"
        if is_mafia:
            result = f"Investigation result: {player_name} IS a member of the Mafia!"
        else:
            result = f"Investigation result: {player_name} is NOT Mafia."

        game.state["last_action"] = {
            "tool": "detective_investigate",
            "args": {"player_name": player_name},
            "target_id": target_id,
            "is_mafia": is_mafia,
        }
        return result

    # -----------------------------------------------------------------------
    # Return all tools grouped by category
    # -----------------------------------------------------------------------

    return {
        "discussion": [make_statement, accuse_player],
        "voting": [cast_vote],
        "mafia": [mafia_kill],
        "doctor": [doctor_protect],
        "detective": [detective_investigate],
    }
