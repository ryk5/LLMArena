from __future__ import annotations

from typing import Any

from llm_arena.core.types import PlayerRole


# ---------------------------------------------------------------------------
# System prompt -- given to every player at the start
# ---------------------------------------------------------------------------

MAFIA_SYSTEM_PROMPT = """\
You are an AI playing a game of Mafia -- a social deduction game.

## Rules Overview

SETUP:
- Players are secretly assigned roles: Mafia, Doctor, Detective, or Villager.
- Mafia members know each other. Town members (Doctor, Detective, Villager) do not know anyone's role.

DAY PHASE:
1. **Discussion**: All living players discuss who they think is Mafia.
   - Make statements, share suspicions, defend yourself, and accuse others.
   - Be strategic: if you are Mafia, deflect suspicion. If you are Town, find the Mafia.
2. **Voting**: All living players vote to eliminate one player.
   - The player with the most votes is eliminated.
   - If there is a tie, no one is eliminated.

NIGHT PHASE:
- Mafia members secretly choose a player to kill.
- The Doctor may protect one player from being killed (cannot protect the same player two nights in a row).
- The Detective may investigate one player to learn if they are Mafia.
- Villagers do nothing at night.

WINNING:
- **Town wins** if all Mafia members are eliminated.
- **Mafia wins** if Mafia members equal or outnumber the remaining town players.

## Important Strategy Tips
- Pay attention to voting patterns and statements for inconsistencies.
- As Mafia, try to blend in and cast suspicion on Town players.
- As Town, share information carefully -- revealing your role can make you a target.
- Use logical reasoning to deduce who is lying.

Play to WIN. Be strategic, persuasive, and analytical.
"""


# ---------------------------------------------------------------------------
# Per-phase view builders
# ---------------------------------------------------------------------------

def build_player_view(
    *,
    player_id: str,
    player_name: str,
    role: PlayerRole,
    phase_name: str,
    round_number: int,
    alive_players: list[dict[str, str]],
    eliminated_players: list[dict[str, str]],
    mafia_teammates: list[str] | None = None,
    extra_context: str = "",
) -> str:
    """
    Build the prompt string that a player sees at the start of a phase.

    Parameters
    ----------
    alive_players : list of {"id": ..., "name": ...} dicts
    eliminated_players : list of {"id": ..., "name": ..., "role": ...} dicts
    mafia_teammates : names of fellow Mafia members (only for Mafia players)
    extra_context : any additional context (e.g. night results summary)
    """
    alive_names = [p["name"] for p in alive_players]
    alive_list = ", ".join(alive_names)

    eliminated_lines = ""
    if eliminated_players:
        lines = []
        for ep in eliminated_players:
            lines.append(f"  - {ep['name']} (was {ep['role']})")
        eliminated_lines = "\n".join(lines)
    else:
        eliminated_lines = "  (none yet)"

    teammate_section = ""
    if mafia_teammates:
        teammate_section = (
            f"\n**Your Mafia teammates**: {', '.join(mafia_teammates)}\n"
            "Coordinate during the night. During the day, do NOT reveal each other."
        )

    view = f"""\
## Round {round_number} -- {phase_name}

**Your identity**: {player_name}
**Your role**: {role.name} ({role.team.capitalize()} team)
{role.description}
{teammate_section}

### Living players ({len(alive_players)})
{alive_list}

### Eliminated players
{eliminated_lines}
"""

    if extra_context:
        view += f"\n{extra_context}\n"

    return view


# ---------------------------------------------------------------------------
# Phase-specific instruction blocks
# ---------------------------------------------------------------------------

def discussion_instructions(player_name: str) -> str:
    """Instructions appended during the DISCUSSION phase."""
    return f"""\
### What to do now
It is the **Day Discussion** phase. Speak your mind to the group.
- Use `make_statement` to share your thoughts, observations, or suspicions.
- Use `accuse_player` to formally accuse someone you believe is Mafia.
- You MUST call exactly one tool. Be strategic about what you reveal.
- Address yourself as {player_name} when speaking.
"""


def voting_instructions() -> str:
    """Instructions appended during the VOTING phase."""
    return """\
### What to do now
It is the **Day Voting** phase. You must vote to eliminate one player.
- Use `cast_vote` with the name of the player you want to eliminate.
- You MUST vote. You cannot abstain.
- The player with the most votes will be eliminated. Ties result in no elimination.
"""


def night_mafia_instructions(alive_names: list[str]) -> str:
    """Instructions for Mafia during the night ACTION phase."""
    targets = ", ".join(alive_names)
    return f"""\
### What to do now
It is **Night**. As Mafia, you must choose a player to kill.
- Use `mafia_kill` with the name of your target.
- Valid targets (living non-Mafia players): {targets}
- Choose strategically -- eliminating the Doctor or Detective weakens the town.
"""


def night_doctor_instructions(
    alive_names: list[str],
    last_protected: str | None,
) -> str:
    """Instructions for the Doctor during the night ACTION phase."""
    targets = ", ".join(alive_names)
    restriction = ""
    if last_protected:
        restriction = (
            f"\n**Restriction**: You protected **{last_protected}** last night. "
            "You CANNOT protect the same player two nights in a row."
        )
    return f"""\
### What to do now
It is **Night**. As the Doctor, you may protect one player from being killed.
- Use `doctor_protect` with the name of the player you want to save.
- Valid targets: {targets}
{restriction}
"""


def night_detective_instructions(alive_names: list[str]) -> str:
    """Instructions for the Detective during the night ACTION phase."""
    targets = ", ".join(alive_names)
    return f"""\
### What to do now
It is **Night**. As the Detective, you may investigate one player.
- Use `detective_investigate` with the name of the player you want to check.
- You will learn whether they are Mafia or not.
- Valid targets: {targets}
"""


def resolution_summary(events: list[str]) -> str:
    """Build a summary of what happened during resolution."""
    body = "\n".join(f"- {e}" for e in events)
    return f"""\
### What happened
{body}
"""
