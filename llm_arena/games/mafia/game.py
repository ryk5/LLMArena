from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from llm_arena.core.game import BaseGame
from llm_arena.core.player import LLMPlayer
from llm_arena.core.types import (
    ActionResult,
    GameOutcome,
    GamePhase,
    PhaseType,
    PlayerRole,
)
from llm_arena.games import register_game
from llm_arena.games.mafia.prompts import (
    MAFIA_SYSTEM_PROMPT,
    build_player_view,
    discussion_instructions,
    night_detective_instructions,
    night_doctor_instructions,
    night_mafia_instructions,
    resolution_summary,
    voting_instructions,
)
from llm_arena.games.mafia.roles import (
    assign_roles,
    get_mafia_members,
    get_players_with_role,
)
from llm_arena.games.mafia.tools import create_mafia_tools


# ---------------------------------------------------------------------------
# Internal phase-cycle state machine
# ---------------------------------------------------------------------------

class _PhaseStep:
    """Tracks where we are in the day/night cycle."""

    DISCUSSION = "discussion"
    VOTING = "voting"
    DAY_RESOLUTION = "day_resolution"
    NIGHT_ACTION = "night_action"
    NIGHT_RESOLUTION = "night_resolution"

    ORDER = [
        DISCUSSION,
        VOTING,
        DAY_RESOLUTION,
        NIGHT_ACTION,
        NIGHT_RESOLUTION,
    ]

    def __init__(self) -> None:
        self._index = -1  # will advance to 0 (DISCUSSION) on first call

    def advance(self) -> str:
        self._index = (self._index + 1) % len(self.ORDER)
        return self.current

    @property
    def current(self) -> str:
        return self.ORDER[self._index]


# ---------------------------------------------------------------------------
# MafiaGame
# ---------------------------------------------------------------------------

@register_game("mafia")
class MafiaGame(BaseGame):
    """
    Mafia -- a multi-player social deduction game.

    Phase cycle per round:
      DISCUSSION -> VOTING -> RESOLUTION (day) ->
      ACTION (night) -> RESOLUTION (night) -> (repeat)
    """

    default_players: int = 7

    # -- setup --------------------------------------------------------------

    async def setup(self) -> None:
        player_ids = [p.player_id for p in self.config.players]

        # Create LLMPlayer instances
        for pinfo in self.config.players:
            self.players[pinfo.player_id] = LLMPlayer(
                info=pinfo,
                client=self.client,
                system_instructions=MAFIA_SYSTEM_PROMPT,
            )

        # Assign roles
        self.roles = assign_roles(player_ids)

        # Build tools (closures over `self`)
        self._tools = create_mafia_tools(self)

        # Initialize game state
        self.state = {
            "alive": set(player_ids),
            "eliminated": [],          # list of {"id", "name", "role", "round", "cause"}
            "votes": {},               # target_id -> count (reset each voting phase)
            "night_actions": {},        # role_action_key -> target_id
            "last_doctor_protect": None,
            "last_action": None,       # populated by tool calls
            "day_events": [],          # summary lines for the current day resolution
            "night_events": [],        # summary lines for the current night resolution
        }

        self._phase_step = _PhaseStep()
        self.round_number = 0

        if self.config.verbose:
            print(f"[Mafia] Roles assigned:")
            for pid, role in self.roles.items():
                print(f"  {self.players[pid].info.name}: {role.name}")

    # -- phase machine ------------------------------------------------------

    async def get_next_phase(self) -> GamePhase:
        step = self._phase_step.advance()

        # Advance round number at the start of each discussion (new day)
        if step == _PhaseStep.DISCUSSION:
            self.round_number += 1

        alive_ids = self._alive_ids()

        if step == _PhaseStep.DISCUSSION:
            return GamePhase(
                phase_type=PhaseType.DISCUSSION,
                round_number=self.round_number,
                description=f"Day {self.round_number} -- Town Discussion",
                active_player_ids=alive_ids,
            )

        if step == _PhaseStep.VOTING:
            # Reset vote tallies
            self.state["votes"] = {}
            return GamePhase(
                phase_type=PhaseType.VOTING,
                round_number=self.round_number,
                description=f"Day {self.round_number} -- Town Vote",
                active_player_ids=alive_ids,
            )

        if step == _PhaseStep.DAY_RESOLUTION:
            self._resolve_day_vote()
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description=f"Day {self.round_number} -- Vote Results",
                active_player_ids=self._alive_ids(),
            )

        if step == _PhaseStep.NIGHT_ACTION:
            # Reset night actions
            self.state["night_actions"] = {}
            night_actors = self._night_active_ids()
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description=f"Night {self.round_number} -- Night Actions",
                active_player_ids=night_actors,
            )

        # NIGHT_RESOLUTION
        self._resolve_night()
        return GamePhase(
            phase_type=PhaseType.RESOLUTION,
            round_number=self.round_number,
            description=f"Night {self.round_number} -- Night Results",
            active_player_ids=self._alive_ids(),
        )

    # -- player view --------------------------------------------------------

    async def get_player_view(self, player_id: str) -> str:
        player = self.players[player_id]
        role = self.roles[player_id]
        step = self._phase_step.current

        # Mafia teammates (names, not including self)
        mafia_teammates: list[str] | None = None
        if role.team == "mafia":
            mafia_ids = get_mafia_members(self.roles)
            mafia_teammates = [
                self.players[mid].info.name
                for mid in mafia_ids
                if mid != player_id and mid in self.state["alive"]
            ]

        # Build extra context based on phase
        extra = ""
        if step == _PhaseStep.DAY_RESOLUTION:
            extra = resolution_summary(self.state.get("day_events", []))
        elif step == _PhaseStep.NIGHT_RESOLUTION:
            extra = resolution_summary(self.state.get("night_events", []))

        # Phase display name
        phase_display = {
            _PhaseStep.DISCUSSION: "Day Discussion",
            _PhaseStep.VOTING: "Day Vote",
            _PhaseStep.DAY_RESOLUTION: "Day Results",
            _PhaseStep.NIGHT_ACTION: "Night",
            _PhaseStep.NIGHT_RESOLUTION: "Night Results",
        }[step]

        view = build_player_view(
            player_id=player_id,
            player_name=player.info.name,
            role=role,
            phase_name=phase_display,
            round_number=self.round_number,
            alive_players=self._alive_player_dicts(),
            eliminated_players=self.state["eliminated"],
            mafia_teammates=mafia_teammates,
            extra_context=extra,
        )

        # Append phase-specific instructions
        if step == _PhaseStep.DISCUSSION:
            view += discussion_instructions(player.info.name)

        elif step == _PhaseStep.VOTING:
            view += voting_instructions()

        elif step == _PhaseStep.NIGHT_ACTION:
            alive_names = self._alive_names()
            if role.name == "Mafia":
                non_mafia_names = [
                    n for n in alive_names
                    if self.roles[self._name_to_id(n)].team != "mafia"
                ]
                view += night_mafia_instructions(non_mafia_names)
            elif role.name == "Doctor":
                last_protected_name = None
                lp = self.state.get("last_doctor_protect")
                if lp and lp in self.players:
                    last_protected_name = self.players[lp].info.name
                view += night_doctor_instructions(alive_names, last_protected_name)
            elif role.name == "Detective":
                other_names = [
                    n for n in alive_names
                    if self._name_to_id(n) != player_id
                ]
                view += night_detective_instructions(other_names)

        return view

    # -- tools for player ---------------------------------------------------

    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        step = self._phase_step.current
        role = self.roles[player_id]

        if step == _PhaseStep.DISCUSSION:
            return self._tools["discussion"]

        if step == _PhaseStep.VOTING:
            return self._tools["voting"]

        if step == _PhaseStep.NIGHT_ACTION:
            if role.name == "Mafia":
                # Only the first alive Mafia member acts (one kill per night)
                first_mafia = self._first_alive_mafia()
                if player_id == first_mafia:
                    return self._tools["mafia"]
                else:
                    return []  # other Mafia members skip
            elif role.name == "Doctor":
                return self._tools["doctor"]
            elif role.name == "Detective":
                return self._tools["detective"]
            else:
                return []  # Villagers do nothing at night

        # RESOLUTION phases -- no tools (the base game's _run_action_phase
        # will skip players with no tools)
        return []

    # -- process action -----------------------------------------------------

    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        last = self.state.get("last_action") or {}
        tool_name = last.get("tool", "unknown")
        args = last.get("args", {})
        error = last.get("error")

        # Determine result text
        if action_result.tool_results:
            # The first tool result text
            result_text = str(action_result.tool_results[0])
        elif action_result.final_output:
            result_text = str(action_result.final_output)
        else:
            result_text = "(no output)"

        success = error is None

        # Build visibility
        visible_to: list[str] | None = None  # default: visible to all
        step = self._phase_step.current

        if step == _PhaseStep.NIGHT_ACTION:
            # Night actions are private
            if tool_name == "mafia_kill":
                # Visible only to Mafia members
                visible_to = [
                    pid for pid in self.state["alive"]
                    if self.roles[pid].team == "mafia"
                ]
            elif tool_name in ("doctor_protect", "detective_investigate"):
                visible_to = [player_id]

        # Reset last action
        self.state["last_action"] = None

        return ActionResult(
            player_id=player_id,
            action_name=tool_name,
            action_args=args,
            result=result_text,
            success=success,
            visible_to=visible_to,
        )

    # -- check game over ----------------------------------------------------

    async def check_game_over(self) -> GameOutcome | None:
        alive = self.state["alive"]
        alive_mafia = [pid for pid in alive if self.roles[pid].team == "mafia"]
        alive_town = [pid for pid in alive if self.roles[pid].team == "town"]

        # Town wins: all Mafia eliminated
        if len(alive_mafia) == 0:
            return self._make_outcome(
                winner_team="town",
                reason="All Mafia members have been eliminated. Town wins!",
            )

        # Mafia wins: Mafia >= Town
        if len(alive_mafia) >= len(alive_town):
            return self._make_outcome(
                winner_team="mafia",
                reason="Mafia equals or outnumbers the town. Mafia wins!",
            )

        # Check max rounds
        if self.round_number >= self.config.max_rounds:
            # Mafia survives by running out the clock
            return self._make_outcome(
                winner_team="mafia",
                reason="Maximum rounds reached. Mafia survives -- Mafia wins!",
            )

        return None

    # -----------------------------------------------------------------------
    # Resolution helpers
    # -----------------------------------------------------------------------

    def _resolve_day_vote(self) -> None:
        """Tally votes and eliminate the top-voted player (or no one on tie)."""
        votes: dict[str, int] = self.state["votes"]
        events: list[str] = []

        if not votes:
            events.append("No votes were cast. No one is eliminated.")
            self.state["day_events"] = events
            return

        # Build vote summary
        for target_id, count in sorted(votes.items(), key=lambda x: -x[1]):
            name = self.players[target_id].info.name
            events.append(f"{name} received {count} vote(s).")

        # Find max
        max_votes = max(votes.values())
        top_voted = [pid for pid, v in votes.items() if v == max_votes]

        if len(top_voted) > 1:
            names = ", ".join(self.players[pid].info.name for pid in top_voted)
            events.append(f"Tie between {names}! No one is eliminated.")
        else:
            eliminated_id = top_voted[0]
            eliminated_name = self.players[eliminated_id].info.name
            eliminated_role = self.roles[eliminated_id].name
            self._eliminate(eliminated_id, cause="voted out")
            events.append(
                f"{eliminated_name} has been eliminated by vote! "
                f"They were a **{eliminated_role}**."
            )

        self.state["day_events"] = events

    def _resolve_night(self) -> None:
        """Resolve all night actions: kill, protect, investigate."""
        actions = self.state["night_actions"]
        events: list[str] = []

        kill_target = actions.get("mafia_kill")
        protect_target = actions.get("doctor_protect")

        # Update doctor protection tracking
        if protect_target:
            self.state["last_doctor_protect"] = protect_target
            protected_name = self.players[protect_target].info.name
            events.append(f"The Doctor chose to protect {protected_name}.")
        else:
            self.state["last_doctor_protect"] = None

        # Resolve kill
        if kill_target:
            target_name = self.players[kill_target].info.name
            if kill_target == protect_target:
                events.append(
                    f"The Mafia targeted {target_name}, but the Doctor saved them! "
                    "No one was killed tonight."
                )
            else:
                killed_role = self.roles[kill_target].name
                self._eliminate(kill_target, cause="killed by Mafia")
                events.append(
                    f"{target_name} was found dead in the morning. "
                    f"They were a **{killed_role}**."
                )
        else:
            events.append("The Mafia did not kill anyone tonight.")

        # Detective investigation is handled in the tool itself (immediate
        # feedback), but we can note it happened.
        investigate_target = actions.get("detective_investigate")
        if investigate_target:
            # This event is kept vague for the public summary
            events.append("The Detective conducted an investigation.")

        self.state["night_events"] = events

    def _eliminate(self, player_id: str, cause: str) -> None:
        """Remove a player from the game."""
        self.state["alive"].discard(player_id)
        role = self.roles[player_id]
        self.state["eliminated"].append({
            "id": player_id,
            "name": self.players[player_id].info.name,
            "role": role.name,
            "round": self.round_number,
            "cause": cause,
        })

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _alive_ids(self) -> list[str]:
        """Ordered list of alive player IDs (preserves config order)."""
        return [
            p.player_id
            for p in self.config.players
            if p.player_id in self.state["alive"]
        ]

    def _alive_names(self) -> list[str]:
        """Ordered list of alive player display names."""
        return [
            self.players[pid].info.name
            for pid in self._alive_ids()
        ]

    def _alive_player_dicts(self) -> list[dict[str, str]]:
        """List of {"id": ..., "name": ...} for alive players."""
        return [
            {"id": pid, "name": self.players[pid].info.name}
            for pid in self._alive_ids()
        ]

    def _night_active_ids(self) -> list[str]:
        """
        Return player IDs that act at night.
        Only the first alive Mafia member, plus Doctor and Detective if alive.
        """
        active: list[str] = []

        # First alive Mafia member (only one acts)
        first_mafia = self._first_alive_mafia()
        if first_mafia:
            active.append(first_mafia)

        # Doctor
        for pid in get_players_with_role(self.roles, "Doctor"):
            if pid in self.state["alive"]:
                active.append(pid)

        # Detective
        for pid in get_players_with_role(self.roles, "Detective"):
            if pid in self.state["alive"]:
                active.append(pid)

        return active

    def _first_alive_mafia(self) -> str | None:
        """Return the first alive Mafia member (by config order)."""
        for p in self.config.players:
            if (
                p.player_id in self.state["alive"]
                and self.roles[p.player_id].team == "mafia"
            ):
                return p.player_id
        return None

    def _name_to_id(self, name: str) -> str:
        """Resolve display name to player ID. Raises if not found."""
        for pid, player in self.players.items():
            if player.info.name == name:
                return pid
        raise ValueError(f"No player with name '{name}'")

    def _make_outcome(self, winner_team: str, reason: str) -> GameOutcome:
        """Build a GameOutcome for the winning team."""
        winner_ids = [
            pid for pid in self.players
            if self.roles[pid].team == winner_team
        ]
        loser_ids = [
            pid for pid in self.players
            if self.roles[pid].team != winner_team
        ]

        return GameOutcome(
            game_id=self.game_id,
            game_type="mafia",
            winner_ids=winner_ids,
            loser_ids=loser_ids,
            metadata={
                "reason": reason,
                "rounds_played": self.round_number,
                "final_alive": list(self.state["alive"]),
                "eliminations": self.state["eliminated"],
                "roles": {
                    pid: self.roles[pid].model_dump()
                    for pid in self.players
                },
            },
            timestamp=datetime.now(timezone.utc),
        )
