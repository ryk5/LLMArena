from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llm_arena.games.secret_hitler.game import SecretHitlerGame


def create_secret_hitler_tools(game: SecretHitlerGame) -> dict[str, Callable]:
    """
    Factory that creates all tool functions for Secret Hitler.

    Each tool captures the ``game`` reference via closure so it can read and
    mutate ``game.state``.  The game engine selects which subset of tools to
    hand to each player depending on the current phase and the player's role.

    Returns a dict mapping tool-name -> callable.
    """

    # ------------------------------------------------------------------
    # Discussion
    # ------------------------------------------------------------------

    def make_statement(statement: str) -> str:
        """Make a public statement during the discussion phase. Share your
        thoughts, suspicions, or arguments with the group.

        Args:
            statement: Your public statement to the group.

        Returns:
            Confirmation that your statement was recorded.
        """
        game.state["last_action"] = {
            "type": "statement",
            "statement": statement,
        }
        return statement

    # ------------------------------------------------------------------
    # Chancellor Nomination
    # ------------------------------------------------------------------

    def nominate_chancellor(player_name: str) -> str:
        """Nominate a player as Chancellor. Only the current President may
        use this tool.

        Args:
            player_name: The name of the player to nominate as Chancellor.

        Returns:
            Result of the nomination attempt.
        """
        # Resolve player name -> id
        target_id = _resolve_player_name(game, player_name)
        if target_id is None:
            game.state["last_action"] = {
                "type": "nominate_chancellor",
                "success": False,
            }
            return (
                f"Invalid nomination: no player named '{player_name}' found. "
                f"Please use the exact player name."
            )

        eligible = game.state.get("eligible_chancellor_ids", [])
        if target_id not in eligible:
            eligible_names = [game.players[pid].info.name for pid in eligible]
            game.state["last_action"] = {
                "type": "nominate_chancellor",
                "success": False,
            }
            return (
                f"'{player_name}' is not eligible for Chancellor. "
                f"Eligible players: {', '.join(eligible_names)}"
            )

        game.state["chancellor_nominee_id"] = target_id
        game.state["last_action"] = {
            "type": "nominate_chancellor",
            "target_id": target_id,
            "success": True,
        }
        nominee_name = game.players[target_id].info.name
        return f"You have nominated {nominee_name} as Chancellor."

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def cast_vote(vote: str) -> str:
        """Cast your vote on the proposed government. Vote 'ja' to approve
        or 'nein' to reject.

        Args:
            vote: Your vote -- must be 'ja' or 'nein'.

        Returns:
            Confirmation of your vote.
        """
        normalized = vote.strip().lower()
        if normalized not in ("ja", "nein"):
            game.state["last_action"] = {
                "type": "cast_vote",
                "success": False,
            }
            return (
                f"Invalid vote '{vote}'. You must vote 'ja' or 'nein'."
            )

        game.state["last_action"] = {
            "type": "cast_vote",
            "vote": normalized,
            "success": True,
        }
        return f"You voted {normalized}."

    # ------------------------------------------------------------------
    # Legislative session -- President discards
    # ------------------------------------------------------------------

    def discard_policy(policy_index: int) -> str:
        """Discard one of the three drawn policies. The remaining two will be
        passed to the Chancellor.

        Args:
            policy_index: Index (0, 1, or 2) of the policy to discard.

        Returns:
            Result of the discard action.
        """
        drawn = game.state.get("drawn_policies", [])
        if not drawn:
            game.state["last_action"] = {"type": "discard_policy", "success": False}
            return "Error: no drawn policies available."

        if policy_index not in (0, 1, 2) or policy_index >= len(drawn):
            game.state["last_action"] = {"type": "discard_policy", "success": False}
            return (
                f"Invalid index {policy_index}. Choose 0, 1, or 2 to discard."
            )

        discarded = drawn[policy_index]
        remaining = [p for i, p in enumerate(drawn) if i != policy_index]

        game.state["discard_pile"].append(discarded)
        game.state["chancellor_policies"] = remaining
        game.state["drawn_policies"] = []  # consumed

        game.state["last_action"] = {
            "type": "discard_policy",
            "discarded": discarded,
            "remaining": remaining,
            "success": True,
        }
        return (
            f"You discarded a {discarded} policy. "
            f"The remaining 2 policies have been passed to the Chancellor."
        )

    # ------------------------------------------------------------------
    # Legislative session -- Chancellor enacts
    # ------------------------------------------------------------------

    def enact_policy(policy_index: int) -> str:
        """Enact one of the two policies passed to you by the President.

        Args:
            policy_index: Index (0 or 1) of the policy to enact.

        Returns:
            Result of the enactment.
        """
        choices = game.state.get("chancellor_policies", [])
        if not choices:
            game.state["last_action"] = {"type": "enact_policy", "success": False}
            return "Error: no policies available to enact."

        if policy_index not in (0, 1) or policy_index >= len(choices):
            game.state["last_action"] = {"type": "enact_policy", "success": False}
            return f"Invalid index {policy_index}. Choose 0 or 1."

        enacted = choices[policy_index]
        discarded = choices[1 - policy_index]

        # Place discarded policy on the discard pile
        game.state["discard_pile"].append(discarded)

        # Enact the chosen policy
        if enacted == "Liberal":
            game.state["liberal_policies"] += 1
        else:
            game.state["fascist_policies"] += 1

        game.state["enacted_policy"] = enacted
        game.state["chancellor_policies"] = []  # consumed

        game.state["last_action"] = {
            "type": "enact_policy",
            "enacted": enacted,
            "success": True,
        }

        # Log for public record
        president_name = game.players[game.state["president_id"]].info.name
        chancellor_name = game.players[game.state["chancellor_id"]].info.name
        game.state.setdefault("policy_log", []).append({
            "round": game.round_number,
            "policy": enacted,
            "president": president_name,
            "chancellor": chancellor_name,
        })

        return f"A {enacted} policy has been enacted!"

    # ------------------------------------------------------------------
    # Presidential powers
    # ------------------------------------------------------------------

    def investigate_player(player_name: str) -> str:
        """Investigate a player's party membership. You will learn whether
        they are Liberal or Fascist. (Hitler's card shows Fascist.)

        Args:
            player_name: The name of the player to investigate.

        Returns:
            The investigated player's party membership.
        """
        target_id = _resolve_player_name(game, player_name)
        if target_id is None:
            game.state["last_action"] = {
                "type": "investigate_player",
                "success": False,
            }
            return f"No player named '{player_name}' found."

        if target_id not in game.state["alive_ids"]:
            game.state["last_action"] = {
                "type": "investigate_player",
                "success": False,
            }
            return f"'{player_name}' is not alive."

        role = game.roles[target_id]
        # Membership card shows team, not specific role
        membership = "Fascist" if role.team == "fascist" else "Liberal"

        game.state.setdefault("investigated_ids", []).append(target_id)
        game.state["last_action"] = {
            "type": "investigate_player",
            "target_id": target_id,
            "membership": membership,
            "success": True,
        }
        return (
            f"You investigated {player_name}. "
            f"Their party membership card says: {membership}."
        )

    def execute_player(player_name: str) -> str:
        """Execute a player, removing them from the game. If you execute
        Hitler, the Liberals win!

        Args:
            player_name: The name of the player to execute.

        Returns:
            Result of the execution.
        """
        target_id = _resolve_player_name(game, player_name)
        if target_id is None:
            game.state["last_action"] = {
                "type": "execute_player",
                "success": False,
            }
            return f"No player named '{player_name}' found."

        if target_id not in game.state["alive_ids"]:
            game.state["last_action"] = {
                "type": "execute_player",
                "success": False,
            }
            return f"'{player_name}' is not alive."

        president_id = game.state["president_id"]
        if target_id == president_id:
            game.state["last_action"] = {
                "type": "execute_player",
                "success": False,
            }
            return "You cannot execute yourself."

        # Remove from alive list
        game.state["alive_ids"].remove(target_id)

        is_hitler = game.roles[target_id].name == "Hitler"
        if is_hitler:
            game.state["hitler_executed"] = True

        game.state["last_action"] = {
            "type": "execute_player",
            "target_id": target_id,
            "is_hitler": is_hitler,
            "success": True,
        }

        result = f"{player_name} has been executed."
        if is_hitler:
            result += " They were Hitler! Liberals win!"
        else:
            result += " They were not Hitler."

        return result

    def choose_next_president(player_name: str) -> str:
        """Choose the next Presidential Candidate via Special Election.

        Args:
            player_name: The name of the player to become the next President.

        Returns:
            Result of the special election choice.
        """
        target_id = _resolve_player_name(game, player_name)
        if target_id is None:
            game.state["last_action"] = {
                "type": "choose_next_president",
                "success": False,
            }
            return f"No player named '{player_name}' found."

        if target_id not in game.state["alive_ids"]:
            game.state["last_action"] = {
                "type": "choose_next_president",
                "success": False,
            }
            return f"'{player_name}' is not alive."

        president_id = game.state["president_id"]
        if target_id == president_id:
            game.state["last_action"] = {
                "type": "choose_next_president",
                "success": False,
            }
            return "You cannot choose yourself."

        game.state["special_election_president_id"] = target_id
        game.state["last_action"] = {
            "type": "choose_next_president",
            "target_id": target_id,
            "success": True,
        }
        target_name = game.players[target_id].info.name
        return f"Special Election: {target_name} will be the next Presidential Candidate."

    # ------------------------------------------------------------------
    # Collect and return
    # ------------------------------------------------------------------

    return {
        "make_statement": make_statement,
        "nominate_chancellor": nominate_chancellor,
        "cast_vote": cast_vote,
        "discard_policy": discard_policy,
        "enact_policy": enact_policy,
        "investigate_player": investigate_player,
        "execute_player": execute_player,
        "choose_next_president": choose_next_president,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_player_name(game: SecretHitlerGame, name: str) -> str | None:
    """
    Resolve a player display-name to their ID.
    Returns None if no match is found.
    Uses case-insensitive comparison and falls back to substring matching.
    """
    name_lower = name.strip().lower()

    # Exact match (case-insensitive)
    for pid, player in game.players.items():
        if player.info.name.lower() == name_lower:
            return pid

    # Substring / fuzzy fallback
    for pid, player in game.players.items():
        if name_lower in player.info.name.lower():
            return pid

    return None
