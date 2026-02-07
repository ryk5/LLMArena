from __future__ import annotations

import random

from llm_arena.core.types import PlayerRole

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

MAFIA_ROLE = PlayerRole(
    name="Mafia",
    team="mafia",
    description=(
        "You are a member of the Mafia. Each night you choose a player to "
        "eliminate. During the day you must blend in with the town and avoid "
        "suspicion. You win when the Mafia equals or outnumbers the remaining "
        "town players."
    ),
    is_hidden=True,
)

DOCTOR_ROLE = PlayerRole(
    name="Doctor",
    team="town",
    description=(
        "You are the Doctor. Each night you may choose one player to protect. "
        "If the Mafia targets that player, they will survive. You cannot "
        "protect the same player two nights in a row. During the day, help the "
        "town identify the Mafia."
    ),
    is_hidden=True,
)

DETECTIVE_ROLE = PlayerRole(
    name="Detective",
    team="town",
    description=(
        "You are the Detective. Each night you may investigate one player to "
        "learn whether they are a member of the Mafia. Use this information "
        "carefully during the day to help the town. Be cautious about revealing "
        "your role -- the Mafia will want to eliminate you."
    ),
    is_hidden=True,
)

VILLAGER_ROLE = PlayerRole(
    name="Villager",
    team="town",
    description=(
        "You are a Villager. You have no special night ability, but your voice "
        "and vote during the day are critical. Pay close attention to what "
        "others say, look for inconsistencies, and help the town identify and "
        "eliminate the Mafia."
    ),
    is_hidden=True,
)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def assign_roles(player_ids: list[str]) -> dict[str, PlayerRole]:
    """
    Assign Mafia roles to a list of player IDs.

    Scaling rules:
      - 5-6 players: 1 Mafia, 1 Doctor, 1 Detective, rest Villagers
      - 7+  players: 2 Mafia, 1 Doctor, 1 Detective, rest Villagers

    Returns a mapping of player_id -> PlayerRole.
    """
    n = len(player_ids)
    if n < 5:
        raise ValueError(f"Mafia requires at least 5 players, got {n}")

    num_mafia = 2 if n >= 7 else 1

    # Build the role list
    role_list: list[PlayerRole] = []
    role_list.extend([MAFIA_ROLE] * num_mafia)
    role_list.append(DOCTOR_ROLE)
    role_list.append(DETECTIVE_ROLE)

    num_villagers = n - len(role_list)
    role_list.extend([VILLAGER_ROLE] * num_villagers)

    # Shuffle and assign
    shuffled_ids = list(player_ids)
    random.shuffle(shuffled_ids)

    assignments: dict[str, PlayerRole] = {}
    for pid, role in zip(shuffled_ids, role_list):
        assignments[pid] = role

    return assignments


def get_mafia_members(roles: dict[str, PlayerRole]) -> list[str]:
    """Return the player IDs of all Mafia members."""
    return [pid for pid, role in roles.items() if role.team == "mafia"]


def get_town_members(roles: dict[str, PlayerRole]) -> list[str]:
    """Return the player IDs of all Town members."""
    return [pid for pid, role in roles.items() if role.team == "town"]


def get_players_with_role(roles: dict[str, PlayerRole], role_name: str) -> list[str]:
    """Return the player IDs that have a specific role name."""
    return [pid for pid, role in roles.items() if role.name == role_name]
