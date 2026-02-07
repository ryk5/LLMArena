from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_arena.games.impostor.game import ImpostorGame

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

IMPOSTOR_SYSTEM_PROMPT = """\
You are an AI playing a social deduction game called "Impostor" (inspired by Among Us) \
aboard a space station. Multiple AI players are competing in this game.

## Game Rules

### Setting
You are on a space station with 8 locations: Cafeteria, Reactor, Electrical, MedBay, \
Navigation, Security, Admin, and Storage. All players start in the Cafeteria.

### Roles
- **Crewmates**: Must complete assigned tasks at specific locations. Win by completing \
all tasks or voting out all impostors.
- **Impostors**: Must eliminate crewmates without being caught. Can fake tasks and kill \
crewmates. Win when the number of living impostors equals the number of living crewmates.

### Gameplay Phases
1. **ACTION phase**: All players move around the station. Crewmates perform tasks. \
Impostors can kill crewmates (if in the same location and cooldown allows). \
Anyone can report a dead body or call an emergency meeting.
2. **DISCUSSION phase**: Triggered by a body report or emergency meeting. All living \
players discuss who they think the impostor is. Share information, make accusations, \
or defend yourself.
3. **VOTING phase**: After discussion, all living players vote to eject someone or skip. \
The player with the most votes is ejected. Ties or majority skip = no ejection.
4. **RESOLUTION phase**: The vote result is announced. If someone was ejected, it is \
revealed whether they were an impostor or crewmate. Then the game returns to ACTION.

### Key Mechanics
- You can only see other players who are at your SAME location.
- Dead players leave bodies at their location until someone reports them.
- Impostors have a kill cooldown of 2 rounds between kills.
- Each player gets 1 emergency meeting call per game.
- Crewmates each have 3 assigned tasks at specific locations. You must be AT the \
location to complete the task.
- Task progress is shared across all crewmates. When all tasks are done, crew wins.

### Strategy Tips
- **As Crewmate**: Complete your tasks efficiently. Watch for suspicious behavior. \
Report bodies you find. During discussions, share what you saw and where.
- **As Impostor**: Fake doing tasks. Kill when alone with a crewmate. Create alibis. \
During discussions, deflect suspicion onto others. Blend in.

Play strategically and pay close attention to what other players say and do!
"""


# ---------------------------------------------------------------------------
# Per-phase prompt builders
# ---------------------------------------------------------------------------

def build_action_prompt(game: ImpostorGame, player_id: str) -> str:
    """Build the prompt shown to a player during the ACTION phase."""
    state = game.state
    player_name = game.players[player_id].info.name
    role = game.roles[player_id]
    location = state["locations"][player_id]
    is_impostor = role.team == "impostor"

    # Who else is at this location (alive only)
    others_here = [
        game.players[pid].info.name
        for pid in state["locations"]
        if pid != player_id
        and state["locations"][pid] == location
        and state["alive"][pid]
    ]

    # Dead bodies at this location
    bodies_here = [
        game.players[pid].info.name
        for pid in state["dead_bodies"]
        if state["dead_bodies"][pid] == location
    ]

    # Build task info
    task_lines = _build_task_info(game, player_id)

    # Overall task progress
    total_tasks = state["total_tasks"]
    completed_tasks = state["completed_tasks"]
    progress_pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

    sections: list[str] = []

    sections.append(f"## ACTION Phase -- Round {game.round_number}")
    sections.append(f"You are **{player_name}**.")
    sections.append(f"Your role: **{role.name}**")

    if is_impostor:
        other_impostors = [
            game.players[pid].info.name
            for pid in state["alive"]
            if pid != player_id
            and game.roles[pid].team == "impostor"
            and state["alive"][pid]
        ]
        if other_impostors:
            sections.append(f"Fellow impostor(s): {', '.join(other_impostors)}")
        else:
            sections.append("You are the sole impostor.")

        cooldown_left = state["kill_cooldowns"].get(player_id, 0)
        if cooldown_left > 0:
            sections.append(f"Kill cooldown: {cooldown_left} round(s) remaining.")
        else:
            sections.append("Kill cooldown: READY -- you can kill this round.")

    sections.append(f"\n### Current Location: {location}")

    if others_here:
        sections.append(f"Players here: {', '.join(others_here)}")
    else:
        sections.append("No other players visible at this location.")

    if bodies_here:
        sections.append(f"DEAD BODIES HERE: {', '.join(bodies_here)}")

    sections.append(f"\n### Task Progress: {completed_tasks}/{total_tasks} ({progress_pct:.0f}%)")
    sections.append(task_lines)

    # Emergency meeting status
    meetings_left = state["emergency_meetings"].get(player_id, 0)
    sections.append(f"\nEmergency meetings remaining: {meetings_left}")

    # Alive players
    alive_names = [
        game.players[pid].info.name
        for pid in state["alive"]
        if state["alive"][pid]
    ]
    sections.append(f"\nAlive players ({len(alive_names)}): {', '.join(alive_names)}")

    # Available actions
    sections.append("\n### Available Actions")
    sections.append("- `move_to(location)`: Move to a different location.")
    sections.append("- `do_task()`: Complete a task at your current location (if you have one here).")
    if bodies_here:
        sections.append("- `report_body()`: Report the dead body you found!")
    if meetings_left > 0:
        sections.append("- `call_emergency_meeting()`: Call an emergency meeting.")
    if is_impostor:
        if others_here and state["kill_cooldowns"].get(player_id, 0) <= 0:
            sections.append("- `kill_player(player_name)`: Kill a crewmate at your location.")

    sections.append("\nChoose ONE action for this round.")

    return "\n".join(sections)


def build_discussion_prompt(game: ImpostorGame, player_id: str) -> str:
    """Build the prompt shown to a player during the DISCUSSION phase."""
    state = game.state
    player_name = game.players[player_id].info.name
    role = game.roles[player_id]
    is_impostor = role.team == "impostor"

    trigger = state.get("meeting_trigger", "unknown")
    trigger_player = state.get("meeting_caller", "unknown")

    sections: list[str] = []

    sections.append(f"## DISCUSSION Phase -- Round {game.round_number}")
    sections.append(f"You are **{player_name}** (Role: **{role.name}**).")

    if trigger == "body_report":
        body_name = state.get("reported_body", "someone")
        sections.append(f"\n**{trigger_player}** reported finding the body of **{body_name}**!")
    elif trigger == "emergency_meeting":
        sections.append(f"\n**{trigger_player}** called an emergency meeting!")

    if is_impostor:
        other_impostors = [
            game.players[pid].info.name
            for pid in state["alive"]
            if pid != player_id
            and game.roles[pid].team == "impostor"
            and state["alive"][pid]
        ]
        if other_impostors:
            sections.append(f"(Secret: Fellow impostor(s): {', '.join(other_impostors)})")

    # Show who is alive
    alive_names = [
        game.players[pid].info.name
        for pid in state["alive"]
        if state["alive"][pid]
    ]
    sections.append(f"\nAlive players ({len(alive_names)}): {', '.join(alive_names)}")

    # Show recently ejected/killed
    if state.get("recently_killed"):
        sections.append(f"Recently killed: {', '.join(state['recently_killed'])}")

    # Show task progress
    total_tasks = state["total_tasks"]
    completed_tasks = state["completed_tasks"]
    progress_pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    sections.append(f"Task progress: {completed_tasks}/{total_tasks} ({progress_pct:.0f}%)")

    # Strategy reminder
    if is_impostor:
        sections.append(
            "\nRemember: You are the Impostor. Deflect suspicion, create doubt, "
            "and try to blend in. Do NOT reveal your role."
        )
    else:
        sections.append(
            "\nRemember: Share what you observed. Where were you? Who did you see? "
            "Work together to identify the impostor."
        )

    sections.append("\nUse `make_statement(statement)` to speak during this discussion.")

    return "\n".join(sections)


def build_voting_prompt(game: ImpostorGame, player_id: str) -> str:
    """Build the prompt shown to a player during the VOTING phase."""
    state = game.state
    player_name = game.players[player_id].info.name
    role = game.roles[player_id]
    is_impostor = role.team == "impostor"

    sections: list[str] = []

    sections.append(f"## VOTING Phase -- Round {game.round_number}")
    sections.append(f"You are **{player_name}** (Role: **{role.name}**).")

    # Alive players who can be voted for
    alive_names = [
        game.players[pid].info.name
        for pid in state["alive"]
        if state["alive"][pid] and pid != player_id
    ]
    sections.append(f"\nPlayers you can vote for: {', '.join(alive_names)}")

    if is_impostor:
        other_impostors = [
            game.players[pid].info.name
            for pid in state["alive"]
            if pid != player_id
            and game.roles[pid].team == "impostor"
            and state["alive"][pid]
        ]
        if other_impostors:
            sections.append(f"(Secret: Fellow impostor(s): {', '.join(other_impostors)} -- avoid voting for them!)")

    sections.append("\nBased on the discussion, cast your vote:")
    sections.append("- `cast_vote(player_name)`: Vote to eject a player.")
    sections.append("- `skip_vote()`: Skip your vote (no ejection preference).")
    sections.append("\nYou MUST use one of these tools. Choose wisely!")

    return "\n".join(sections)


def build_resolution_prompt(game: ImpostorGame, player_id: str) -> str:
    """Build the prompt shown during the RESOLUTION phase (informational)."""
    state = game.state

    sections: list[str] = []
    sections.append(f"## RESOLUTION -- Round {game.round_number}")

    vote_result = state.get("vote_result", {})
    ejected_name = vote_result.get("ejected_name")
    ejected_role = vote_result.get("ejected_role")
    vote_counts = vote_result.get("vote_counts", {})
    skips = vote_result.get("skips", 0)

    if vote_counts:
        vote_summary = ", ".join(f"{name}: {count}" for name, count in vote_counts.items())
        sections.append(f"Vote tally: {vote_summary} | Skips: {skips}")

    if ejected_name:
        sections.append(f"\n**{ejected_name}** was ejected!")
        sections.append(f"They were a **{ejected_role}**.")
    else:
        sections.append("\nNo one was ejected (tie or majority skip).")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_task_info(game: ImpostorGame, player_id: str) -> str:
    """Build task list display for a player."""
    role = game.roles[player_id]
    tasks = game.state["tasks"].get(player_id, [])

    if role.team == "impostor":
        # Impostors see fake task guidance
        return (
            "Your tasks (FAKE -- you cannot actually complete these):\n"
            "  Move around and pretend to do tasks to avoid suspicion."
        )

    if not tasks:
        return "All your tasks are complete!"

    lines = ["Your assigned tasks:"]
    for i, t in enumerate(tasks, 1):
        status = "DONE" if t["completed"] else "pending"
        lines.append(f"  {i}. [{status}] {t['task']} at {t['location']}")

    return "\n".join(lines)
