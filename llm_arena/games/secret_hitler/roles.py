from __future__ import annotations

import random

from llm_arena.core.types import PlayerRole

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

LIBERAL_ROLE = PlayerRole(
    name="Liberal",
    team="liberal",
    description=(
        "You are a Liberal. Your goal is to enact 5 Liberal policies or to "
        "identify and assassinate Hitler. You do not know anyone else's role. "
        "During discussions, try to figure out who the Fascists are and vote "
        "against suspicious Chancellor nominations. Trust is your greatest "
        "weapon -- and your greatest vulnerability."
    ),
    is_hidden=True,
)

FASCIST_ROLE = PlayerRole(
    name="Fascist",
    team="fascist",
    description=(
        "You are a Fascist. Your goal is to enact 6 Fascist policies or to get "
        "Hitler elected as Chancellor after 3 or more Fascist policies have been "
        "enacted. You know who your fellow Fascists are, including Hitler. "
        "Sow confusion, cast suspicion on Liberals, and protect Hitler's identity "
        "while working to get Fascist policies enacted."
    ),
    is_hidden=True,
)

HITLER_ROLE = PlayerRole(
    name="Hitler",
    team="fascist",
    description=(
        "You are Hitler. Your goal is the same as the Fascists: enact 6 Fascist "
        "policies, or get yourself elected as Chancellor after 3 or more Fascist "
        "policies are enacted. You must appear trustworthy and Liberal to avoid "
        "being assassinated. Play carefully and try to get elected Chancellor "
        "when the time is right."
    ),
    is_hidden=True,
)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _role_counts(num_players: int) -> tuple[int, int]:
    """
    Return (num_regular_fascists, num_liberals) for a given player count.
    Hitler is always 1 additional Fascist-team member.

    5-6 players: 1 Fascist + Hitler = 2 fascist-team, rest Liberal
    7-8 players: 2 Fascists + Hitler = 3 fascist-team, rest Liberal
    9-10 players: 3 Fascists + Hitler = 4 fascist-team, rest Liberal
    """
    if num_players <= 6:
        num_fascists = 1
    elif num_players <= 8:
        num_fascists = 2
    else:
        num_fascists = 3
    num_liberals = num_players - num_fascists - 1  # -1 for Hitler
    return num_fascists, num_liberals


def assign_roles(player_ids: list[str]) -> dict[str, PlayerRole]:
    """
    Assign Secret Hitler roles to a list of player IDs.

    Returns a mapping of player_id -> PlayerRole.
    """
    n = len(player_ids)
    if n < 5:
        raise ValueError(f"Secret Hitler requires at least 5 players, got {n}")
    if n > 10:
        raise ValueError(f"Secret Hitler supports at most 10 players, got {n}")

    num_fascists, num_liberals = _role_counts(n)

    role_list: list[PlayerRole] = []
    role_list.append(HITLER_ROLE)
    role_list.extend([FASCIST_ROLE] * num_fascists)
    role_list.extend([LIBERAL_ROLE] * num_liberals)

    shuffled_ids = list(player_ids)
    random.shuffle(shuffled_ids)

    assignments: dict[str, PlayerRole] = {}
    for pid, role in zip(shuffled_ids, role_list):
        assignments[pid] = role

    return assignments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_fascist_team(roles: dict[str, PlayerRole]) -> list[str]:
    """Return player IDs of all Fascist-team members (Fascists + Hitler)."""
    return [pid for pid, role in roles.items() if role.team == "fascist"]


def get_liberals(roles: dict[str, PlayerRole]) -> list[str]:
    """Return player IDs of all Liberal players."""
    return [pid for pid, role in roles.items() if role.team == "liberal"]


def get_hitler(roles: dict[str, PlayerRole]) -> str:
    """Return the player ID of Hitler."""
    for pid, role in roles.items():
        if role.name == "Hitler":
            return pid
    raise ValueError("No Hitler found in role assignments")


def get_regular_fascists(roles: dict[str, PlayerRole]) -> list[str]:
    """Return player IDs of Fascists who are NOT Hitler."""
    return [pid for pid, role in roles.items()
            if role.team == "fascist" and role.name != "Hitler"]


def hitler_knows_fascists(num_players: int) -> bool:
    """In 5-6 player games, Hitler knows who the other Fascist is."""
    return num_players <= 6
