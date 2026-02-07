from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Callable

from llm_arena.core.game import BaseGame
from llm_arena.core.player import LLMPlayer
from llm_arena.core.types import (
    ActionResult,
    GameConfig,
    GameOutcome,
    GamePhase,
    PhaseType,
)
from llm_arena.games import register_game
from llm_arena.games.secret_hitler.prompts import (
    SYSTEM_PROMPT,
    chancellor_enact_prompt,
    discussion_prompt,
    execution_prompt,
    investigation_prompt,
    nomination_prompt,
    peek_prompt,
    president_discard_prompt,
    special_election_prompt,
    vote_result_summary,
    voting_prompt,
)
from llm_arena.games.secret_hitler.roles import (
    assign_roles,
    get_fascist_team,
    get_hitler,
    get_liberals,
)
from llm_arena.games.secret_hitler.tools import create_secret_hitler_tools

# ---------------------------------------------------------------------------
# Sub-phase identifiers used in state["sub_phase"]
# ---------------------------------------------------------------------------

PHASE_DISCUSSION = "discussion"
PHASE_NOMINATION = "nomination"
PHASE_VOTING = "voting"
PHASE_VOTE_RESULT = "vote_result"
PHASE_PRESIDENT_DISCARD = "president_discard"
PHASE_CHANCELLOR_ENACT = "chancellor_enact"
PHASE_POLICY_ENACTED = "policy_enacted"
PHASE_POWER_INVESTIGATE = "power_investigate"
PHASE_POWER_EXECUTION = "power_execution"
PHASE_POWER_SPECIAL_ELECTION = "power_special_election"
PHASE_POWER_PEEK = "power_peek"
PHASE_ROUND_END = "round_end"


# ---------------------------------------------------------------------------
# Presidential-power tables indexed by fascist-policy count
# ---------------------------------------------------------------------------

def _power_table(num_players: int) -> dict[int, str | None]:
    """
    Return a mapping of fascist-policy-count -> presidential power (or None).
    The power triggers *after* the Nth fascist policy is enacted.
    """
    if num_players <= 6:
        return {
            1: None,
            2: None,
            3: "peek",
            4: "execution",
            5: "execution",
        }
    elif num_players <= 8:
        return {
            1: None,
            2: "investigate",
            3: "special_election",
            4: "execution",
            5: "execution",
        }
    else:
        return {
            1: "investigate",
            2: "investigate",
            3: "special_election",
            4: "execution",
            5: "execution",
        }


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

@register_game("secret_hitler")
class SecretHitlerGame(BaseGame):
    """
    Secret Hitler implementation for LLMArena.

    Supports 5-10 players. The game is divided into rounds; each round
    progresses through a series of sub-phases managed via ``state["sub_phase"]``.
    """

    default_players: int = 7

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self._tools: dict[str, Callable] = {}
        self._player_order: list[str] = []  # clockwise seat order (alive)
        self._full_player_order: list[str] = []  # original seat order

    # ---------------------------------------------------------------
    # Setup
    # ---------------------------------------------------------------

    async def setup(self) -> None:
        """Initialise players, roles, policy deck, and game state."""
        # Create LLMPlayer objects
        for pinfo in self.config.players:
            player = LLMPlayer(
                info=pinfo,
                client=self.client,
                system_instructions=SYSTEM_PROMPT,
            )
            self.players[pinfo.player_id] = player

        player_ids = list(self.players.keys())

        # Assign roles
        self.roles = assign_roles(player_ids)

        # Build seat order (random shuffle for fairness)
        self._full_player_order = list(player_ids)
        random.shuffle(self._full_player_order)
        self._player_order = list(self._full_player_order)

        # Build tool functions (closures referencing self)
        self._tools = create_secret_hitler_tools(self)

        # Build & shuffle the policy deck
        deck = ["Liberal"] * 6 + ["Fascist"] * 11
        random.shuffle(deck)

        # Initialise state
        self.state = {
            # Policy tracking
            "liberal_policies": 0,
            "fascist_policies": 0,
            "policy_deck": deck,
            "discard_pile": [],
            "policy_log": [],
            # Government
            "president_index": -1,  # index into _full_player_order; first advance will land on 0
            "president_id": None,
            "chancellor_nominee_id": None,
            "chancellor_id": None,
            "prev_president_id": None,
            "prev_chancellor_id": None,
            "term_limited_ids": [],
            "eligible_chancellor_ids": [],
            # Election tracker
            "election_tracker": 0,
            # Player liveness
            "alive_ids": list(self._player_order),
            # Sub-phase sequencing
            "sub_phase": PHASE_DISCUSSION,
            # Voting
            "votes": {},
            # Legislative session
            "drawn_policies": [],
            "chancellor_policies": [],
            "enacted_policy": None,
            # Powers
            "investigated_ids": [],
            "special_election_president_id": None,
            "peeked_policies": [],
            "hitler_executed": False,
            # Last action placeholder
            "last_action": {},
        }

        # Set the first president
        self._advance_president()

        if self.config.verbose:
            self._print_setup_summary()

    # ---------------------------------------------------------------
    # Phase sequencing
    # ---------------------------------------------------------------

    async def get_next_phase(self) -> GamePhase:
        """Return the next GamePhase based on the current sub-phase."""
        sub = self.state["sub_phase"]
        alive = self.state["alive_ids"]

        if sub == PHASE_DISCUSSION:
            return GamePhase(
                phase_type=PhaseType.DISCUSSION,
                round_number=self.round_number,
                description=f"Round {self.round_number} discussion",
                active_player_ids=list(alive),
            )

        if sub == PHASE_NOMINATION:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="President nominates a Chancellor",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_VOTING:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="All players vote on the proposed government",
                active_player_ids=list(alive),
            )

        if sub == PHASE_VOTE_RESULT:
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description="Tallying votes",
                active_player_ids=[],
            )

        if sub == PHASE_PRESIDENT_DISCARD:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="President discards one policy",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_CHANCELLOR_ENACT:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="Chancellor enacts one policy",
                active_player_ids=[self.state["chancellor_id"]],
            )

        if sub == PHASE_POLICY_ENACTED:
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description="Policy enacted -- checking for presidential powers",
                active_player_ids=[],
            )

        if sub == PHASE_POWER_INVESTIGATE:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="Presidential Power: Investigate Loyalty",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_POWER_EXECUTION:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="Presidential Power: Execution",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_POWER_SPECIAL_ELECTION:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="Presidential Power: Special Election",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_POWER_PEEK:
            return GamePhase(
                phase_type=PhaseType.ACTION,
                round_number=self.round_number,
                description="Presidential Power: Policy Peek",
                active_player_ids=[self.state["president_id"]],
            )

        if sub == PHASE_ROUND_END:
            # Transition to next round
            self._begin_next_round()
            return await self.get_next_phase()

        # Fallback: game over
        return GamePhase(
            phase_type=PhaseType.GAME_OVER,
            round_number=self.round_number,
            description="Game over",
            active_player_ids=[],
        )

    # ---------------------------------------------------------------
    # Player view (prompt)
    # ---------------------------------------------------------------

    async def get_player_view(self, player_id: str) -> str:
        sub = self.state["sub_phase"]

        if sub == PHASE_DISCUSSION:
            return discussion_prompt(self, player_id)
        if sub == PHASE_NOMINATION:
            return nomination_prompt(self, player_id)
        if sub == PHASE_VOTING:
            return voting_prompt(self, player_id)
        if sub == PHASE_PRESIDENT_DISCARD:
            return president_discard_prompt(self, player_id)
        if sub == PHASE_CHANCELLOR_ENACT:
            return chancellor_enact_prompt(self, player_id)
        if sub == PHASE_POWER_INVESTIGATE:
            return investigation_prompt(self, player_id)
        if sub == PHASE_POWER_EXECUTION:
            return execution_prompt(self, player_id)
        if sub == PHASE_POWER_SPECIAL_ELECTION:
            return special_election_prompt(self, player_id)
        if sub == PHASE_POWER_PEEK:
            return peek_prompt(self, player_id)

        return discussion_prompt(self, player_id)

    # ---------------------------------------------------------------
    # Tools per player
    # ---------------------------------------------------------------

    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        sub = self.state["sub_phase"]

        if sub == PHASE_DISCUSSION:
            return [self._tools["make_statement"]]

        if sub == PHASE_NOMINATION:
            return [self._tools["nominate_chancellor"]]

        if sub == PHASE_VOTING:
            return [self._tools["cast_vote"]]

        if sub == PHASE_PRESIDENT_DISCARD:
            return [self._tools["discard_policy"]]

        if sub == PHASE_CHANCELLOR_ENACT:
            return [self._tools["enact_policy"]]

        if sub == PHASE_POWER_INVESTIGATE:
            return [self._tools["investigate_player"]]

        if sub == PHASE_POWER_EXECUTION:
            return [self._tools["execute_player"]]

        if sub == PHASE_POWER_SPECIAL_ELECTION:
            return [self._tools["choose_next_president"]]

        if sub == PHASE_POWER_PEEK:
            # Peek is informational; player acknowledges with a statement
            return [self._tools["make_statement"]]

        return []

    # ---------------------------------------------------------------
    # Process action
    # ---------------------------------------------------------------

    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        """
        Called after a player's LLM turn.  Reads ``state["last_action"]``
        (set by the tool function) and produces an ActionResult.
        """
        last = self.state.get("last_action", {})
        action_type = last.get("type", "unknown")
        success = last.get("success", True)

        # Extract a human-readable result string
        if action_type == "statement":
            result_text = last.get("statement", "")
        elif action_type == "nominate_chancellor":
            target_id = last.get("target_id")
            if target_id and success:
                result_text = f"Nominated {self.players[target_id].info.name} as Chancellor"
            else:
                result_text = "Nomination failed"
        elif action_type == "cast_vote":
            result_text = f"Voted {last.get('vote', '?')}"
        elif action_type == "discard_policy":
            result_text = f"Discarded a policy"
        elif action_type == "enact_policy":
            result_text = f"Enacted {last.get('enacted', '?')} policy"
        elif action_type == "investigate_player":
            target_id = last.get("target_id")
            if target_id and success:
                result_text = f"Investigated {self.players[target_id].info.name}"
            else:
                result_text = "Investigation failed"
        elif action_type == "execute_player":
            target_id = last.get("target_id")
            if target_id and success:
                result_text = f"Executed {self.players[target_id].info.name}"
            else:
                result_text = "Execution failed"
        elif action_type == "choose_next_president":
            target_id = last.get("target_id")
            if target_id and success:
                result_text = f"Chose {self.players[target_id].info.name} as next President"
            else:
                result_text = "Special election choice failed"
        else:
            # Fallback: try to extract text from the LLM result
            result_text = str(getattr(action_result, "final_output", ""))[:200]

        # Determine visibility
        visible_to: list[str] | None = None  # default: visible to all
        if action_type == "cast_vote":
            # Votes are secret until tallied
            visible_to = [player_id]
        elif action_type == "discard_policy":
            visible_to = [player_id]
        elif action_type == "investigate_player":
            visible_to = [player_id]

        action = ActionResult(
            player_id=player_id,
            action_name=action_type,
            action_args=last,
            result=result_text,
            success=success,
            visible_to=visible_to,
        )

        # ---- Trigger sub-phase transitions based on the action ----
        await self._handle_post_action(player_id, last)

        return action

    # ---------------------------------------------------------------
    # Post-action sub-phase transitions
    # ---------------------------------------------------------------

    async def _handle_post_action(self, player_id: str, last: dict) -> None:
        """Advance the sub-phase state machine after an action."""
        sub = self.state["sub_phase"]
        action_type = last.get("type", "")
        success = last.get("success", True)

        # -- Nomination --
        if sub == PHASE_NOMINATION and action_type == "nominate_chancellor":
            if success:
                self.state["sub_phase"] = PHASE_VOTING
                self.state["votes"] = {}
            # If nomination failed the president must try again (phase stays)

        # -- Voting --
        elif sub == PHASE_VOTING and action_type == "cast_vote":
            if success:
                self.state["votes"][player_id] = last["vote"]
                # Check if all alive players have voted
                alive = self.state["alive_ids"]
                if len(self.state["votes"]) >= len(alive):
                    self.state["sub_phase"] = PHASE_VOTE_RESULT

        # -- President discard --
        elif sub == PHASE_PRESIDENT_DISCARD and action_type == "discard_policy":
            if success:
                self.state["sub_phase"] = PHASE_CHANCELLOR_ENACT

        # -- Chancellor enact --
        elif sub == PHASE_CHANCELLOR_ENACT and action_type == "enact_policy":
            if success:
                self.state["sub_phase"] = PHASE_POLICY_ENACTED

        # -- Presidential powers --
        elif sub == PHASE_POWER_INVESTIGATE and action_type == "investigate_player":
            if success:
                self.state["sub_phase"] = PHASE_ROUND_END

        elif sub == PHASE_POWER_EXECUTION and action_type == "execute_player":
            if success:
                self.state["sub_phase"] = PHASE_ROUND_END

        elif sub == PHASE_POWER_SPECIAL_ELECTION and action_type == "choose_next_president":
            if success:
                self.state["sub_phase"] = PHASE_ROUND_END

        elif sub == PHASE_POWER_PEEK and action_type == "statement":
            self.state["sub_phase"] = PHASE_ROUND_END

    # ---------------------------------------------------------------
    # Resolution phases (no player actions -- game logic only)
    # ---------------------------------------------------------------

    async def _run_discussion_phase(self):
        """Run the discussion, then advance the sub-phase to nomination."""
        await super()._run_discussion_phase()
        self.state["sub_phase"] = PHASE_NOMINATION

    async def _run_action_phase(self):
        """
        Override to handle resolution phases that have no active players.
        For normal action phases, delegate to the base class.
        """
        sub = self.state["sub_phase"]

        if sub == PHASE_VOTE_RESULT:
            await self._resolve_vote()
            return

        if sub == PHASE_POLICY_ENACTED:
            await self._resolve_policy_enacted()
            return

        # Normal action phase
        await super()._run_action_phase()

    async def _resolve_vote(self) -> None:
        """Tally votes and decide whether the government is formed."""
        votes = self.state["votes"]
        ja_count = sum(1 for v in votes.values() if v == "ja")
        nein_count = sum(1 for v in votes.values() if v == "nein")
        passed = ja_count > nein_count

        summary = vote_result_summary(self, votes, passed)
        if self.config.verbose:
            print(f"  [VOTE] {summary}")

        # Log the vote as a resolution action
        self.action_log.append(ActionResult(
            player_id="system",
            action_name="vote_result",
            action_args={"votes": votes, "passed": passed},
            result=summary,
            success=True,
        ))
        self.transcript.log_action(self.action_log[-1])

        if passed:
            # Government formed
            chancellor_id = self.state["chancellor_nominee_id"]
            self.state["chancellor_id"] = chancellor_id
            self.state["election_tracker"] = 0

            # Check Hitler-chancellor win condition
            if self.state["fascist_policies"] >= 3:
                hitler_id = get_hitler(self.roles)
                if chancellor_id == hitler_id:
                    self.state["hitler_elected_chancellor"] = True
                    self.state["sub_phase"] = PHASE_ROUND_END
                    return

            # Begin legislative session
            self._draw_policies()
            self.state["sub_phase"] = PHASE_PRESIDENT_DISCARD

        else:
            # Failed election
            self.state["election_tracker"] += 1

            if self.state["election_tracker"] >= 3:
                # Chaos: auto-enact top policy
                self._auto_enact_top_policy()
                self.state["election_tracker"] = 0
                # Reset term limits after chaos
                self.state["prev_president_id"] = None
                self.state["prev_chancellor_id"] = None
                self.state["term_limited_ids"] = []
                # Check if the auto-enacted policy triggers a game end
                self.state["sub_phase"] = PHASE_POLICY_ENACTED
            else:
                # Move to next round (next president)
                self.state["sub_phase"] = PHASE_ROUND_END

    async def _resolve_policy_enacted(self) -> None:
        """Check for presidential powers after a policy is enacted."""
        enacted = self.state.get("enacted_policy")

        if enacted == "Fascist":
            fascist_count = self.state["fascist_policies"]
            num_players = len(self._full_player_order)
            powers = _power_table(num_players)
            power = powers.get(fascist_count)

            if power == "investigate":
                self.state["sub_phase"] = PHASE_POWER_INVESTIGATE
                return
            elif power == "execution":
                self.state["sub_phase"] = PHASE_POWER_EXECUTION
                return
            elif power == "special_election":
                self.state["sub_phase"] = PHASE_POWER_SPECIAL_ELECTION
                return
            elif power == "peek":
                # Show the president the top 3 policies
                self._ensure_deck_size(3)
                self.state["peeked_policies"] = list(
                    self.state["policy_deck"][:3]
                )
                self.state["sub_phase"] = PHASE_POWER_PEEK
                return

        # No power -- end round
        self.state["sub_phase"] = PHASE_ROUND_END

    # ---------------------------------------------------------------
    # Check game over
    # ---------------------------------------------------------------

    async def check_game_over(self) -> GameOutcome | None:
        liberal_ids = get_liberals(self.roles)
        fascist_ids = get_fascist_team(self.roles)

        # Liberal victory: 5 liberal policies
        if self.state["liberal_policies"] >= 5:
            return self._make_outcome(
                winner_team="liberal",
                reason="5 Liberal policies enacted",
            )

        # Liberal victory: Hitler executed
        if self.state.get("hitler_executed"):
            return self._make_outcome(
                winner_team="liberal",
                reason="Hitler was executed",
            )

        # Fascist victory: 6 fascist policies
        if self.state["fascist_policies"] >= 6:
            return self._make_outcome(
                winner_team="fascist",
                reason="6 Fascist policies enacted",
            )

        # Fascist victory: Hitler elected chancellor after 3+ fascist policies
        if self.state.get("hitler_elected_chancellor"):
            return self._make_outcome(
                winner_team="fascist",
                reason="Hitler elected Chancellor after 3+ Fascist policies",
            )

        # Safety valve: max rounds
        if self.round_number >= self.config.max_rounds:
            return self._make_outcome(
                winner_team="liberal",
                reason=f"Max rounds ({self.config.max_rounds}) reached -- draw goes to Liberals",
            )

        return None

    def _make_outcome(self, winner_team: str, reason: str) -> GameOutcome:
        liberal_ids = get_liberals(self.roles)
        fascist_ids = get_fascist_team(self.roles)

        if winner_team == "liberal":
            winners, losers = liberal_ids, fascist_ids
        else:
            winners, losers = fascist_ids, liberal_ids

        return GameOutcome(
            game_id=self.game_id,
            game_type="secret_hitler",
            winner_ids=winners,
            loser_ids=losers,
            metadata={
                "reason": reason,
                "liberal_policies": self.state["liberal_policies"],
                "fascist_policies": self.state["fascist_policies"],
                "rounds_played": self.round_number,
                "roles": {
                    pid: {"name": role.name, "team": role.team}
                    for pid, role in self.roles.items()
                },
            },
            timestamp=datetime.now(timezone.utc),
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _advance_president(self) -> None:
        """
        Set the next president according to the rotation (skipping dead
        players), or honour a special-election override.
        """
        alive = self.state["alive_ids"]

        # Special election override
        special = self.state.get("special_election_president_id")
        if special and special in alive:
            self.state["president_id"] = special
            self.state["special_election_president_id"] = None
            # Don't advance president_index -- after special election the
            # normal rotation continues from where it left off.
        else:
            # Normal clockwise rotation
            order = self._full_player_order
            idx = self.state["president_index"]
            for _ in range(len(order)):
                idx = (idx + 1) % len(order)
                candidate = order[idx]
                if candidate in alive:
                    self.state["president_index"] = idx
                    self.state["president_id"] = candidate
                    break

        self._compute_term_limits()
        self._compute_eligible_chancellors()

    def _compute_term_limits(self) -> None:
        """Determine who is term-limited this round."""
        alive = self.state["alive_ids"]
        num_alive = len(alive)
        term_limited: list[str] = []

        prev_chan = self.state.get("prev_chancellor_id")
        prev_pres = self.state.get("prev_president_id")

        # The previous Chancellor is always term-limited
        if prev_chan and prev_chan in alive:
            term_limited.append(prev_chan)

        # The previous President is term-limited only if >5 alive players
        if num_alive > 5 and prev_pres and prev_pres in alive:
            term_limited.append(prev_pres)

        self.state["term_limited_ids"] = term_limited

    def _compute_eligible_chancellors(self) -> None:
        """Build the list of player IDs eligible to be Chancellor."""
        alive = self.state["alive_ids"]
        president_id = self.state["president_id"]
        term_limited = self.state["term_limited_ids"]

        eligible = [
            pid for pid in alive
            if pid != president_id and pid not in term_limited
        ]
        self.state["eligible_chancellor_ids"] = eligible

    def _draw_policies(self) -> None:
        """President draws 3 policies from the deck."""
        self._ensure_deck_size(3)
        deck = self.state["policy_deck"]
        drawn = [deck.pop(0) for _ in range(3)]
        self.state["drawn_policies"] = drawn

    def _ensure_deck_size(self, needed: int) -> None:
        """If the deck is too small, shuffle the discard pile back in."""
        deck = self.state["policy_deck"]
        if len(deck) < needed:
            deck.extend(self.state["discard_pile"])
            self.state["discard_pile"] = []
            random.shuffle(deck)

    def _auto_enact_top_policy(self) -> None:
        """Enact the top policy from the deck (chaos / 3 failed elections)."""
        self._ensure_deck_size(1)
        policy = self.state["policy_deck"].pop(0)
        if policy == "Liberal":
            self.state["liberal_policies"] += 1
        else:
            self.state["fascist_policies"] += 1

        self.state["enacted_policy"] = policy

        # Log
        self.state.setdefault("policy_log", []).append({
            "round": self.round_number,
            "policy": policy,
            "president": "chaos",
            "chancellor": "chaos",
        })

        if self.config.verbose:
            print(f"  [CHAOS] 3 failed elections! A {policy} policy was auto-enacted.")

        self.action_log.append(ActionResult(
            player_id="system",
            action_name="chaos_enact",
            action_args={},
            result=f"Chaos: {policy} policy auto-enacted after 3 failed elections",
            success=True,
        ))
        self.transcript.log_action(self.action_log[-1])

    def _begin_next_round(self) -> None:
        """Clean up the current round and prepare for the next one."""
        # Record previous government for term limits (only if election passed)
        if self.state.get("chancellor_id"):
            self.state["prev_president_id"] = self.state["president_id"]
            self.state["prev_chancellor_id"] = self.state["chancellor_id"]

        # Reset per-round state
        self.state["chancellor_nominee_id"] = None
        self.state["chancellor_id"] = None
        self.state["votes"] = {}
        self.state["drawn_policies"] = []
        self.state["chancellor_policies"] = []
        self.state["enacted_policy"] = None
        self.state["peeked_policies"] = []
        self.state["last_action"] = {}

        self.round_number += 1
        self._advance_president()
        self.state["sub_phase"] = PHASE_DISCUSSION

    def _print_setup_summary(self) -> None:
        """Print role assignments and seat order (for verbose mode)."""
        print(f"\n=== Secret Hitler Game {self.game_id} ===")
        print(f"Players: {len(self.players)}")
        print("Seat order and roles:")
        for pid in self._full_player_order:
            name = self.players[pid].info.name
            role = self.roles[pid]
            print(f"  {name}: {role.name} ({role.team})")
        print(f"First President: {self.players[self.state['president_id']].info.name}")
        deck = self.state["policy_deck"]
        lib_count = deck.count("Liberal")
        fas_count = deck.count("Fascist")
        print(f"Policy deck: {lib_count} Liberal, {fas_count} Fascist\n")
