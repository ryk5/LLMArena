from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from llm_arena.core.game import BaseGame
from llm_arena.core.player import LLMPlayer
from llm_arena.core.types import (
    ActionResult,
    GameOutcome,
    GamePhase,
    PhaseType,
)
from llm_arena.games import register_game
from llm_arena.games.impostor.prompts import (
    IMPOSTOR_SYSTEM_PROMPT,
    build_action_prompt,
    build_discussion_prompt,
    build_resolution_prompt,
    build_voting_prompt,
)
from llm_arena.games.impostor.roles import (
    assign_roles,
    generate_tasks,
    get_crewmates,
    get_impostors,
)
from llm_arena.games.impostor.tools import create_impostor_tools


@register_game("impostor")
class ImpostorGame(BaseGame):
    """
    Impostor (Among Us-style) social deduction game for the LLM Arena.

    Players are assigned roles as Crewmates or Impostors on a space station.
    Crewmates complete tasks and try to identify impostors through discussion
    and voting. Impostors try to eliminate crewmates without being caught.

    Win conditions:
      - Crewmates win: all tasks completed OR all impostors ejected.
      - Impostors win: living impostors >= living crewmates.

    Phase cycle:
      ACTION -> (body report or meeting) -> DISCUSSION -> VOTING -> RESOLUTION -> ACTION
    """

    default_players: int = 6

    # Maximum action rounds before the game is force-ended.
    MAX_ACTION_ROUNDS: int = 30

    def __init__(self, config):
        super().__init__(config)
        self._tools: dict[str, list[Callable]] = {}
        # Tracks which sub-phase we are in within a round cycle.
        self._phase_queue: list[PhaseType] = []
        # Whether we are in the middle of processing actions (to detect meeting triggers)
        self._action_round_count: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Initialize the game: players, roles, tasks, locations."""
        n = len(self.config.players)
        if n < 4 or n > 10:
            raise ValueError(
                f"Impostor requires 4-10 players, got {n}"
            )

        # Create LLMPlayer instances
        for info in self.config.players:
            self.players[info.player_id] = LLMPlayer(
                info=info,
                client=self.client,
                system_instructions=IMPOSTOR_SYSTEM_PROMPT,
            )

        # Assign roles
        player_ids = list(self.players.keys())
        self.roles = assign_roles(player_ids)

        # Generate tasks for crewmates
        tasks = generate_tasks(player_ids, self.roles)

        # Count total tasks across all crewmates
        total_tasks = sum(
            len(t_list)
            for pid, t_list in tasks.items()
            if self.roles[pid].team == "crew"
        )

        # Initialize game state
        self.state = {
            # Player locations -- everyone starts in Cafeteria
            "locations": {pid: "Cafeteria" for pid in player_ids},
            # Alive status
            "alive": {pid: True for pid in player_ids},
            # Tasks per player
            "tasks": tasks,
            # Task progress
            "total_tasks": total_tasks,
            "completed_tasks": 0,
            # Kill cooldowns (impostor_id -> rounds remaining)
            "kill_cooldowns": {pid: 0 for pid in get_impostors(self.roles)},
            # Dead bodies waiting to be reported (victim_id -> location)
            "dead_bodies": {},
            # Emergency meetings remaining per player (1 each)
            "emergency_meetings": {pid: 1 for pid in player_ids},
            # Meeting trigger info (set when body reported or meeting called)
            "meeting_triggered": False,
            "meeting_trigger": None,
            "meeting_caller": None,
            "reported_body": None,
            # Votes for current voting round
            "votes": {},
            # Vote result for resolution display
            "vote_result": {},
            # Recently killed (for discussion context)
            "recently_killed": [],
            # Current player being processed (set during action/discussion)
            "current_player_id": None,
            # Last action taken (set by tool closures)
            "last_action": None,
            # Ejected players log
            "ejected": [],
        }

        # Create tool closures
        self._tools = create_impostor_tools(self)

        # Start with ACTION phase queued
        self._phase_queue = [PhaseType.ACTION]
        self._action_round_count = 0

        if self.config.verbose:
            impostor_names = [
                self.players[pid].info.name for pid in get_impostors(self.roles)
            ]
            crew_names = [
                self.players[pid].info.name for pid in get_crewmates(self.roles)
            ]
            print(f"Impostor game {self.game_id} initialized with {n} players.")
            print(f"  Impostors: {', '.join(impostor_names)}")
            print(f"  Crewmates: {', '.join(crew_names)}")
            print(f"  Total tasks: {total_tasks}")

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    async def get_next_phase(self) -> GamePhase:
        """Determine the next game phase."""
        # Check for game-ending conditions first
        outcome = await self.check_game_over()
        if outcome is not None:
            return GamePhase(
                phase_type=PhaseType.GAME_OVER,
                round_number=self.round_number,
                description="The game is over.",
                active_player_ids=[],
            )

        # If phase queue is empty, default to next ACTION round
        if not self._phase_queue:
            self._phase_queue = [PhaseType.ACTION]

        next_phase_type = self._phase_queue.pop(0)

        alive_ids = [pid for pid in self.state["alive"] if self.state["alive"][pid]]

        if next_phase_type == PhaseType.ACTION:
            self._action_round_count += 1
            self.round_number = self._action_round_count

            # Decrement kill cooldowns at the start of each action round
            for imp_id in self.state["kill_cooldowns"]:
                if self.state["kill_cooldowns"][imp_id] > 0:
                    self.state["kill_cooldowns"][imp_id] -= 1

            # Clear meeting state
            self.state["meeting_triggered"] = False
            self.state["recently_killed"] = []

            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description=f"Action Round {self._action_round_count}: Players move, do tasks, or take actions.",
                active_player_ids=alive_ids,
            )

        elif next_phase_type == PhaseType.DISCUSSION:
            return GamePhase(
                phase_type=PhaseType.DISCUSSION,
                round_number=self.round_number,
                description="Discussion phase: Players discuss who they suspect.",
                active_player_ids=alive_ids,
            )

        elif next_phase_type == PhaseType.VOTING:
            # Reset votes
            self.state["votes"] = {}
            return GamePhase(
                phase_type=PhaseType.VOTING,
                round_number=self.round_number,
                description="Voting phase: Players vote on who to eject.",
                active_player_ids=alive_ids,
            )

        elif next_phase_type == PhaseType.RESOLUTION:
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description="Resolution: Vote results are tallied.",
                active_player_ids=alive_ids,
            )

        # Fallback
        return GamePhase(
            phase_type=PhaseType.ACTION,
            round_number=self.round_number,
            description="Continuing play.",
            active_player_ids=alive_ids,
        )

    # ------------------------------------------------------------------
    # Player view
    # ------------------------------------------------------------------

    async def get_player_view(self, player_id: str) -> str:
        """Build the prompt string for the given player in the current phase."""
        if self.phase is None:
            return "Waiting for game to start..."

        phase_type = self.phase.phase_type

        if phase_type == PhaseType.ACTION:
            return build_action_prompt(self, player_id)
        elif phase_type == PhaseType.DISCUSSION:
            return build_discussion_prompt(self, player_id)
        elif phase_type == PhaseType.VOTING:
            return build_voting_prompt(self, player_id)
        elif phase_type == PhaseType.RESOLUTION:
            return build_resolution_prompt(self, player_id)
        else:
            return f"Phase: {phase_type.value}"

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        """Return the appropriate tools for this player in the current phase."""
        if self.phase is None:
            return []

        phase_type = self.phase.phase_type

        if phase_type == PhaseType.ACTION:
            role = self.roles[player_id]
            if role.team == "impostor":
                return self._tools.get("action_impostor", [])
            else:
                return self._tools.get("action_crew", [])

        elif phase_type == PhaseType.DISCUSSION:
            return self._tools.get("discussion", [])

        elif phase_type == PhaseType.VOTING:
            return self._tools.get("voting", [])

        elif phase_type == PhaseType.RESOLUTION:
            # Resolution is informational; no tools needed.
            # But framework expects tools for action phases, so return empty.
            return []

        return []

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        """
        Process a player's action result.

        The tool closures in tools.py directly mutate game state and set
        self.state["last_action"]. We read that to build the ActionResult.
        """
        last = self.state.get("last_action")

        if last is None:
            # Try to get tool info from action_result
            tool_name = "unknown"
            tools_called = getattr(action_result, "tools_called", [])
            if tools_called:
                tool_name = tools_called[-1]

            return ActionResult(
                player_id=player_id,
                action_name=tool_name,
                action_args={},
                result="No action was recorded.",
                success=False,
                visible_to=None,
            )

        # Build visibility list based on action type
        visible_to = None  # Default: visible to all
        tool_name = last.get("tool", "unknown")

        # Kill results should only be visible to the killer
        if tool_name == "kill_player" and last.get("success"):
            visible_to = [player_id]

        # Discussion statements are visible to all alive players
        # Votes are visible to the voter only (results shown in resolution)
        if tool_name in ("cast_vote", "skip_vote"):
            visible_to = [player_id]

        result = ActionResult(
            player_id=player_id,
            action_name=tool_name,
            action_args=last.get("args", {}),
            result=last.get("result", ""),
            success=last.get("success", False),
            visible_to=visible_to,
        )

        # Clear last action
        self.state["last_action"] = None

        return result

    # ------------------------------------------------------------------
    # Overridden phase runners
    # ------------------------------------------------------------------

    async def _run_action_phase(self):
        """
        Run the action phase: each alive player takes one action.

        After all players have acted, check if a meeting was triggered.
        If so, queue DISCUSSION -> VOTING -> RESOLUTION phases.
        """
        for player_id in self.phase.active_player_ids:
            # Skip dead players (in case someone was killed this round)
            if not self.state["alive"].get(player_id, False):
                continue

            # Set current player context for tool closures
            self.state["current_player_id"] = player_id
            self.state["last_action"] = None

            player = self.players[player_id]
            view = await self.get_player_view(player_id)
            tools = await self.get_tools_for_player(player_id)

            if not tools:
                continue

            action_result = await player.take_action(
                game_prompt=view,
                tools=tools,
                phase=self.phase,
            )

            processed = await self.process_action(player_id, action_result)
            self.action_log.append(processed)
            self.transcript.log_action(processed)

            if self.config.verbose:
                print(f"  [{player.info.name}] {processed.action_name}: {processed.result}")

            # Check if a meeting was triggered by this action
            if self.state.get("meeting_triggered"):
                # Stop processing remaining players and go to discussion
                break

            # Check for immediate game-over after a kill
            outcome = await self.check_game_over()
            if outcome is not None:
                return

        # After action phase, decide what happens next
        if self.state.get("meeting_triggered"):
            self._phase_queue = [
                PhaseType.DISCUSSION,
                PhaseType.VOTING,
                PhaseType.RESOLUTION,
            ]
        else:
            # No meeting triggered; queue next action round
            self._phase_queue = [PhaseType.ACTION]

    async def _run_discussion_phase(self):
        """
        Discussion phase: each alive player makes a statement.
        Uses the base class pattern but sets current_player_id for tool closures.
        """
        discussion_history: list[dict[str, str]] = []

        for player_id in self.phase.active_player_ids:
            if not self.state["alive"].get(player_id, False):
                continue

            self.state["current_player_id"] = player_id
            self.state["last_action"] = None

            player = self.players[player_id]
            view = await self.get_player_view(player_id)

            if discussion_history:
                view += "\n\n## Discussion so far:\n" + "\n".join(
                    f"**{d['name']}**: {d['statement']}" for d in discussion_history
                )

            tools = await self.get_tools_for_player(player_id)
            if not tools:
                continue

            action_result = await player.take_action(
                game_prompt=view,
                tools=tools,
                phase=self.phase,
            )

            processed = await self.process_action(player_id, action_result)
            self.action_log.append(processed)
            self.transcript.log_action(processed)

            discussion_history.append(
                {"name": player.info.name, "statement": processed.result}
            )

            if self.config.verbose:
                print(f"  [{player.info.name}] says: {processed.result}")

    async def _run_voting_phase(self):
        """Run the voting phase: each alive player votes or skips."""
        for player_id in self.phase.active_player_ids:
            if not self.state["alive"].get(player_id, False):
                continue

            self.state["current_player_id"] = player_id
            self.state["last_action"] = None

            player = self.players[player_id]
            view = await self.get_player_view(player_id)
            tools = await self.get_tools_for_player(player_id)

            if not tools:
                continue

            action_result = await player.take_action(
                game_prompt=view,
                tools=tools,
                phase=self.phase,
            )

            processed = await self.process_action(player_id, action_result)
            self.action_log.append(processed)
            self.transcript.log_action(processed)

            if self.config.verbose:
                print(f"  [{player.info.name}] {processed.action_name}: {processed.result}")

    async def _run_resolution_phase(self):
        """
        Tally votes, eject the most-voted player (if any), and announce results.
        """
        votes = self.state.get("votes", {})
        alive_ids = [pid for pid in self.state["alive"] if self.state["alive"][pid]]

        # Count votes
        vote_counts: Counter = Counter()
        skips = 0

        for voter_id, target in votes.items():
            if target == "skip":
                skips += 1
            else:
                vote_counts[target] += 1

        # Determine outcome
        ejected_id = None
        ejected_name = None
        ejected_role = None

        if vote_counts:
            # Find the player(s) with the most votes
            max_votes = vote_counts.most_common(1)[0][1]
            top_voted = [pid for pid, count in vote_counts.items() if count == max_votes]

            # Only eject if there is a single leader AND they have more votes than skips
            if len(top_voted) == 1 and max_votes > skips:
                ejected_id = top_voted[0]
                ejected_name = self.players[ejected_id].info.name
                ejected_role = self.roles[ejected_id].name

                # Eject the player
                self.state["alive"][ejected_id] = False
                self.state["ejected"].append(ejected_id)

                # Remove from dead_bodies if somehow there (shouldn't be, but safety)
                self.state["dead_bodies"].pop(ejected_id, None)

        # Build human-readable vote counts with names
        vote_count_display = {}
        for pid, count in vote_counts.items():
            vote_count_display[self.players[pid].info.name] = count

        # Store result for resolution prompt
        self.state["vote_result"] = {
            "ejected_id": ejected_id,
            "ejected_name": ejected_name,
            "ejected_role": ejected_role,
            "vote_counts": vote_count_display,
            "skips": skips,
        }

        if self.config.verbose:
            if vote_count_display:
                tally = ", ".join(
                    f"{name}: {cnt}" for name, cnt in vote_count_display.items()
                )
                print(f"  Vote tally: {tally} | Skips: {skips}")
            if ejected_name:
                print(f"  {ejected_name} was ejected! They were a {ejected_role}.")
            else:
                print("  No one was ejected.")

        # Clear meeting state
        self.state["meeting_triggered"] = False
        self.state["meeting_trigger"] = None
        self.state["meeting_caller"] = None
        self.state["reported_body"] = None
        self.state["votes"] = {}

    # ------------------------------------------------------------------
    # Main game loop override
    # ------------------------------------------------------------------

    async def run(self) -> GameOutcome:
        """
        Main game loop, customized for Impostor's multi-sub-phase structure.

        The base class run() only handles DISCUSSION and ACTION via
        _run_discussion_phase and _run_action_phase. We need VOTING and
        RESOLUTION sub-phases as well, so we override the main loop.
        """
        await self.setup()
        self.transcript.log_game_start(self.game_id, self.config)

        while True:
            self.phase = await self.get_next_phase()
            self.transcript.log_phase(self.phase)

            if self.phase.phase_type == PhaseType.GAME_OVER:
                break

            if self.phase.phase_type == PhaseType.ACTION:
                await self._run_action_phase()
            elif self.phase.phase_type == PhaseType.DISCUSSION:
                await self._run_discussion_phase()
            elif self.phase.phase_type == PhaseType.VOTING:
                await self._run_voting_phase()
            elif self.phase.phase_type == PhaseType.RESOLUTION:
                await self._run_resolution_phase()

            outcome = await self.check_game_over()
            if outcome is not None:
                self.transcript.log_game_end(outcome)
                return outcome

            # Safety: prevent infinite loops
            if self._action_round_count > self.MAX_ACTION_ROUNDS:
                break

        outcome = await self.check_game_over()
        if outcome is None:
            # Force a draw if we hit max rounds
            outcome = self._build_draw_outcome()
        self.transcript.log_game_end(outcome)
        return outcome

    # ------------------------------------------------------------------
    # Game over check
    # ------------------------------------------------------------------

    async def check_game_over(self) -> GameOutcome | None:
        """
        Check if the game has ended.

        Win conditions:
          - Crewmates win: all tasks completed OR all impostors ejected/dead.
          - Impostors win: living impostors >= living crewmates.
        """
        alive_impostors = [
            pid for pid in get_impostors(self.roles)
            if self.state["alive"].get(pid, False)
        ]
        alive_crewmates = [
            pid for pid in get_crewmates(self.roles)
            if self.state["alive"].get(pid, False)
        ]

        impostor_ids = get_impostors(self.roles)
        crewmate_ids = get_crewmates(self.roles)

        # Crew wins: all impostors eliminated
        if len(alive_impostors) == 0:
            return GameOutcome(
                game_id=self.game_id,
                game_type="impostor",
                winner_ids=crewmate_ids,
                loser_ids=impostor_ids,
                metadata={
                    "termination": "all_impostors_ejected",
                    "rounds_played": self._action_round_count,
                    "tasks_completed": self.state["completed_tasks"],
                    "total_tasks": self.state["total_tasks"],
                },
                timestamp=datetime.now(timezone.utc),
            )

        # Crew wins: all tasks completed
        if self.state["completed_tasks"] >= self.state["total_tasks"]:
            return GameOutcome(
                game_id=self.game_id,
                game_type="impostor",
                winner_ids=crewmate_ids,
                loser_ids=impostor_ids,
                metadata={
                    "termination": "all_tasks_completed",
                    "rounds_played": self._action_round_count,
                    "tasks_completed": self.state["completed_tasks"],
                    "total_tasks": self.state["total_tasks"],
                },
                timestamp=datetime.now(timezone.utc),
            )

        # Impostor wins: living impostors >= living crewmates
        if len(alive_impostors) >= len(alive_crewmates):
            return GameOutcome(
                game_id=self.game_id,
                game_type="impostor",
                winner_ids=impostor_ids,
                loser_ids=crewmate_ids,
                metadata={
                    "termination": "impostor_majority",
                    "rounds_played": self._action_round_count,
                    "alive_impostors": len(alive_impostors),
                    "alive_crewmates": len(alive_crewmates),
                    "tasks_completed": self.state["completed_tasks"],
                    "total_tasks": self.state["total_tasks"],
                },
                timestamp=datetime.now(timezone.utc),
            )

        return None

    def _build_draw_outcome(self) -> GameOutcome:
        """Build a draw outcome when max rounds are reached."""
        impostor_ids = get_impostors(self.roles)
        crewmate_ids = get_crewmates(self.roles)

        return GameOutcome(
            game_id=self.game_id,
            game_type="impostor",
            winner_ids=[],
            loser_ids=impostor_ids + crewmate_ids,
            metadata={
                "termination": "max_rounds_reached",
                "rounds_played": self._action_round_count,
                "tasks_completed": self.state["completed_tasks"],
                "total_tasks": self.state["total_tasks"],
            },
            timestamp=datetime.now(timezone.utc),
        )
