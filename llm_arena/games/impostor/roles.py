from __future__ import annotations

import random

from llm_arena.core.types import PlayerRole

# ---------------------------------------------------------------------------
# Location and task definitions
# ---------------------------------------------------------------------------

LOCATIONS = [
    "Cafeteria",
    "Reactor",
    "Electrical",
    "MedBay",
    "Navigation",
    "Security",
    "Admin",
    "Storage",
]

# Tasks available at each location. Each task is a short description.
LOCATION_TASKS: dict[str, list[str]] = {
    "Cafeteria": ["Empty garbage", "Fix wiring", "Accept diverted power"],
    "Reactor": ["Start reactor sequence", "Unlock manifolds", "Divert power"],
    "Electrical": ["Fix wiring", "Calibrate distributor", "Divert power"],
    "MedBay": ["Submit scan", "Inspect sample", "Sort samples"],
    "Navigation": ["Chart course", "Stabilize steering", "Fix wiring"],
    "Security": ["Fix wiring", "Sort files", "Swipe card"],
    "Admin": ["Swipe card", "Upload data", "Fix wiring"],
    "Storage": ["Empty garbage", "Fuel engines", "Fix wiring"],
}

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

IMPOSTOR_ROLE = PlayerRole(
    name="Impostor",
    team="impostor",
    description=(
        "You are an IMPOSTOR aboard the space station. Your goal is to "
        "eliminate crewmates without being discovered. You can fake tasks, "
        "kill crewmates when you are alone with them, and manipulate "
        "discussions to avoid suspicion. You win when the number of living "
        "impostors equals the number of living crewmates."
    ),
    is_hidden=True,
)

CREWMATE_ROLE = PlayerRole(
    name="Crewmate",
    team="crew",
    description=(
        "You are a CREWMATE aboard the space station. Complete your assigned "
        "tasks by traveling to the correct locations and performing them. "
        "Stay alert for suspicious behavior -- if you find a dead body, "
        "report it immediately. During discussions, work with other crewmates "
        "to identify and vote out the impostor(s). You win when all tasks "
        "are completed or all impostors are ejected."
    ),
    is_hidden=True,
)


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

def _impostor_count(num_players: int) -> int:
    """Determine how many impostors to assign based on player count."""
    if num_players <= 5:
        return 1
    elif num_players <= 6:
        # 6 players: randomly 1 or 2
        return random.choice([1, 2])
    elif num_players <= 8:
        return 2
    else:
        # 9-10 players
        return 2


def assign_roles(player_ids: list[str]) -> dict[str, PlayerRole]:
    """
    Assign Impostor game roles to a list of player IDs.

    Scaling rules:
      - 4-5 players: 1 Impostor
      - 6 players:   1-2 Impostors (random)
      - 7-8 players: 2 Impostors
      - 9-10 players: 2 Impostors

    Returns a mapping of player_id -> PlayerRole.
    """
    n = len(player_ids)
    if n < 4:
        raise ValueError(f"Impostor requires at least 4 players, got {n}")
    if n > 10:
        raise ValueError(f"Impostor supports at most 10 players, got {n}")

    num_impostors = _impostor_count(n)

    shuffled_ids = list(player_ids)
    random.shuffle(shuffled_ids)

    assignments: dict[str, PlayerRole] = {}
    for i, pid in enumerate(shuffled_ids):
        if i < num_impostors:
            assignments[pid] = IMPOSTOR_ROLE
        else:
            assignments[pid] = CREWMATE_ROLE

    return assignments


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

def generate_tasks(player_ids: list[str], roles: dict[str, PlayerRole]) -> dict[str, list[dict]]:
    """
    Generate 3 random location-task pairs for each crewmate.

    Each task is a dict with:
      - location: str (which location the task is at)
      - task: str (description of the task)
      - completed: bool (starts False)

    Impostors receive an empty task list (they fake tasks).

    Returns a mapping of player_id -> list of task dicts.
    """
    all_tasks: dict[str, list[dict]] = {}

    for pid in player_ids:
        if roles[pid].team == "impostor":
            all_tasks[pid] = []
            continue

        # Pick 3 distinct location-task pairs for this crewmate
        possible_pairs: list[tuple[str, str]] = []
        for location, tasks in LOCATION_TASKS.items():
            for task in tasks:
                possible_pairs.append((location, task))

        chosen = random.sample(possible_pairs, k=3)
        all_tasks[pid] = [
            {"location": loc, "task": task_name, "completed": False}
            for loc, task_name in chosen
        ]

    return all_tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_impostors(roles: dict[str, PlayerRole]) -> list[str]:
    """Return the player IDs of all Impostors."""
    return [pid for pid, role in roles.items() if role.team == "impostor"]


def get_crewmates(roles: dict[str, PlayerRole]) -> list[str]:
    """Return the player IDs of all Crewmates."""
    return [pid for pid, role in roles.items() if role.team == "crew"]
