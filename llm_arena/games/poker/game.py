from __future__ import annotations

import random
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
from llm_arena.games.poker.hand_eval import (
    compare_hands,
    evaluate_hand,
    hand_rank_name,
)
from llm_arena.games.poker.prompts import POKER_SYSTEM_PROMPT, get_turn_prompt
from llm_arena.games.poker.tools import create_poker_tools

# ---- Constants ----
SUITS = "cdhs"
RANKS = "23456789TJQKA"
FULL_DECK = [r + s for s in SUITS for r in RANKS]

STREETS = ["pre_flop", "flop", "turn", "river", "showdown"]

DEFAULT_STARTING_CHIPS = 1000
DEFAULT_SMALL_BLIND = 10
DEFAULT_BIG_BLIND = 20


@register_game("poker")
class PokerGame(BaseGame):
    """
    Texas Hold'em poker for 2-6 LLM players.

    The game consists of multiple hands.  Players are eliminated when they
    run out of chips.  The last player standing wins.
    """

    default_players: int = 2

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        starting_chips = int(
            self.config.options.get("starting_chips", DEFAULT_STARTING_CHIPS)
        )
        small_blind = int(
            self.config.options.get("small_blind", DEFAULT_SMALL_BLIND)
        )
        big_blind = int(
            self.config.options.get("big_blind", DEFAULT_BIG_BLIND)
        )

        # Create players
        for pinfo in self.config.players:
            player = LLMPlayer(
                info=pinfo,
                client=self.client,
                system_instructions=POKER_SYSTEM_PROMPT,
            )
            self.players[pinfo.player_id] = player
            self.roles[pinfo.player_id] = PlayerRole(
                name="player",
                team="none",
                description="Poker player",
                is_hidden=False,
            )

        player_ids = list(self.players.keys())

        self.state.update(
            {
                # Global state across hands
                "chips": {pid: starting_chips for pid in player_ids},
                "player_ids": player_ids,  # all players (including eliminated)
                "eliminated": set(),
                "dealer_index": 0,  # index into active_seat_ids
                "hand_number": 0,
                "small_blind": small_blind,
                "big_blind": big_blind,
                # Per-hand state (initialised in _start_hand)
                "hand_active": False,
                "street": "pre_flop",
                "deck": [],
                "hole_cards": {},
                "community_cards": [],
                "pot": 0,
                "current_bet": 0,
                "min_raise": big_blind * 2,
                "round_bets": {},
                "folded_players": set(),
                "all_in_players": set(),
                "acted_this_round": set(),
                "last_raiser": None,
                "hand_history": [],
                "betting_started": False,
                # Per-action bookkeeping
                "last_action": None,
                "acting_player_id": None,
                # Phase control
                "needs_new_hand": True,
                "action_queue": [],  # player ids waiting to act
            }
        )

    # ------------------------------------------------------------------
    # Hand lifecycle helpers
    # ------------------------------------------------------------------

    def _active_seat_ids(self) -> list[str]:
        """Players still in the tournament (have chips or are in a hand)."""
        return [
            pid
            for pid in self.state["player_ids"]
            if pid not in self.state["eliminated"]
        ]

    def _hand_active_ids(self) -> list[str]:
        """Players still in the current hand (not folded, not eliminated)."""
        return [
            pid
            for pid in self._active_seat_ids()
            if pid not in self.state["folded_players"]
        ]

    def _can_act_ids(self) -> list[str]:
        """Players who can still take betting actions (not folded, not all-in)."""
        return [
            pid
            for pid in self._hand_active_ids()
            if pid not in self.state["all_in_players"]
        ]

    def _start_hand(self) -> None:
        """Set up a fresh hand: shuffle, deal hole cards, post blinds."""
        self.state["hand_number"] += 1
        self.state["hand_active"] = True
        self.state["street"] = "pre_flop"
        self.state["community_cards"] = []
        self.state["pot"] = 0
        self.state["current_bet"] = 0
        self.state["min_raise"] = self.state["big_blind"] * 2
        self.state["round_bets"] = {}
        self.state["folded_players"] = set()
        self.state["all_in_players"] = set()
        self.state["acted_this_round"] = set()
        self.state["last_raiser"] = None
        self.state["hand_history"] = []
        self.state["betting_started"] = False
        self.state["last_action"] = None
        self.state["action_queue"] = []
        self.state["needs_new_hand"] = False

        seats = self._active_seat_ids()

        # Shuffle and deal
        deck = list(FULL_DECK)
        random.shuffle(deck)
        self.state["deck"] = deck

        hole: dict[str, list[str]] = {}
        for pid in seats:
            hole[pid] = [deck.pop(), deck.pop()]
        self.state["hole_cards"] = hole

        # Post blinds
        n = len(seats)
        dealer_idx = self.state["dealer_index"] % n
        sb_idx = (dealer_idx + 1) % n
        bb_idx = (dealer_idx + 2) % n

        # For heads-up (2 players), dealer posts small blind
        if n == 2:
            sb_idx = dealer_idx
            bb_idx = (dealer_idx + 1) % n

        sb_id = seats[sb_idx]
        bb_id = seats[bb_idx]
        self.state["sb_id"] = sb_id
        self.state["bb_id"] = bb_id
        self.state["dealer_id"] = seats[dealer_idx]

        sb_amount = min(self.state["small_blind"], self.state["chips"][sb_id])
        bb_amount = min(self.state["big_blind"], self.state["chips"][bb_id])

        self.state["chips"][sb_id] -= sb_amount
        self.state["chips"][bb_id] -= bb_amount
        self.state["pot"] += sb_amount + bb_amount
        self.state["round_bets"][sb_id] = sb_amount
        self.state["round_bets"][bb_id] = bb_amount
        self.state["current_bet"] = bb_amount
        self.state["min_raise"] = bb_amount * 2

        if self.state["chips"][sb_id] == 0:
            self.state["all_in_players"].add(sb_id)
        if self.state["chips"][bb_id] == 0:
            self.state["all_in_players"].add(bb_id)

        self.state["hand_history"].append(
            f"  {self._player_name(sb_id)} posts small blind {sb_amount}"
        )
        self.state["hand_history"].append(
            f"  {self._player_name(bb_id)} posts big blind {bb_amount}"
        )

        # Build the pre-flop action order
        self._build_action_queue_preflop(seats, dealer_idx, sb_idx, bb_idx)

    def _build_action_queue_preflop(
        self,
        seats: list[str],
        dealer_idx: int,
        sb_idx: int,
        bb_idx: int,
    ) -> None:
        """Pre-flop action starts left of BB and wraps around."""
        n = len(seats)
        start = (bb_idx + 1) % n
        order: list[str] = []
        for i in range(n):
            pid = seats[(start + i) % n]
            if pid not in self.state["all_in_players"]:
                order.append(pid)
        self.state["action_queue"] = order

    def _build_action_queue_postflop(self) -> None:
        """Post-flop action starts left of dealer."""
        seats = self._active_seat_ids()
        n = len(seats)
        dealer_id = self.state["dealer_id"]
        try:
            dealer_idx = seats.index(dealer_id)
        except ValueError:
            dealer_idx = 0
        start = (dealer_idx + 1) % n
        order: list[str] = []
        for i in range(n):
            pid = seats[(start + i) % n]
            if (
                pid not in self.state["folded_players"]
                and pid not in self.state["all_in_players"]
            ):
                order.append(pid)
        self.state["action_queue"] = order

    def _advance_street(self) -> None:
        """Deal the next community cards and reset betting state."""
        street = self.state["street"]
        idx = STREETS.index(street)
        if idx >= len(STREETS) - 1:
            return
        next_street = STREETS[idx + 1]
        self.state["street"] = next_street

        deck = self.state["deck"]
        if next_street == "flop":
            self.state["community_cards"].extend([deck.pop(), deck.pop(), deck.pop()])
            self.state["hand_history"].append(
                f"  --- Flop: {' '.join(self.state['community_cards'])} ---"
            )
        elif next_street == "turn":
            self.state["community_cards"].append(deck.pop())
            self.state["hand_history"].append(
                f"  --- Turn: {' '.join(self.state['community_cards'])} ---"
            )
        elif next_street == "river":
            self.state["community_cards"].append(deck.pop())
            self.state["hand_history"].append(
                f"  --- River: {' '.join(self.state['community_cards'])} ---"
            )
        elif next_street == "showdown":
            pass  # handled in _resolve_showdown

        # Reset per-round betting state
        self.state["current_bet"] = 0
        self.state["min_raise"] = self.state["big_blind"]
        self.state["round_bets"] = {}
        self.state["acted_this_round"] = set()
        self.state["last_raiser"] = None
        self.state["betting_started"] = False

        if next_street != "showdown":
            self._build_action_queue_postflop()

    def _is_betting_complete(self) -> bool:
        """Check whether the current betting round is finished."""
        can_act = self._can_act_ids()

        # If 0 or 1 players can act, betting is done
        if len(can_act) <= 1:
            # But if there is one player who hasn't acted and no bet to match,
            # they still need a chance to act (unless already acted)
            if len(can_act) == 1:
                pid = can_act[0]
                cur_bet = self.state["current_bet"]
                my_bet = self.state["round_bets"].get(pid, 0)
                if pid not in self.state["acted_this_round"] and cur_bet <= my_bet:
                    # They haven't acted yet and no bet to match -- not done yet
                    # unless everyone else is all-in or folded and no action queue
                    if not self.state["action_queue"]:
                        return True
                    return False
            return True

        # Everyone who can act must have acted, and bets must be equal
        cur_bet = self.state["current_bet"]
        for pid in can_act:
            my_bet = self.state["round_bets"].get(pid, 0)
            if my_bet < cur_bet:
                return False
            if pid not in self.state["acted_this_round"]:
                return False

        return True

    def _is_hand_over(self) -> bool:
        """True if only one player remains (all others folded) or showdown reached."""
        if self.state["street"] == "showdown":
            return True
        active = self._hand_active_ids()
        if len(active) <= 1:
            return True
        return False

    def _resolve_hand(self) -> None:
        """Determine the winner and award the pot."""
        active = self._hand_active_ids()
        pot = self.state["pot"]
        community = self.state["community_cards"]

        if len(active) == 1:
            winner_id = active[0]
            self.state["chips"][winner_id] += pot
            self.state["hand_history"].append(
                f"  {self._player_name(winner_id)} wins {pot} chips (all others folded)."
            )
        elif len(active) > 1:
            # Showdown -- fill in remaining community cards if needed
            deck = self.state["deck"]
            while len(community) < 5:
                community.append(deck.pop())
            self.state["community_cards"] = community

            # Evaluate hands
            hands: list[tuple[int, list[int]]] = []
            hand_players: list[str] = []
            for pid in active:
                hole = self.state["hole_cards"][pid]
                ev = evaluate_hand(hole, community)
                hands.append(ev)
                hand_players.append(pid)

            winner_indices = compare_hands(hands)

            # Log showdown details
            self.state["hand_history"].append(
                f"  --- Showdown: {' '.join(community)} ---"
            )
            for i, pid in enumerate(hand_players):
                hole = self.state["hole_cards"][pid]
                rank, _ = hands[i]
                self.state["hand_history"].append(
                    f"  {self._player_name(pid)} shows {' '.join(hole)} -- {hand_rank_name(rank)}"
                )

            # Split pot among winners
            split = pot // len(winner_indices)
            remainder = pot % len(winner_indices)
            for idx in winner_indices:
                winner_id = hand_players[idx]
                award = split + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                self.state["chips"][winner_id] += award
                self.state["hand_history"].append(
                    f"  {self._player_name(winner_id)} wins {award} chips "
                    f"with {hand_rank_name(hands[idx][0])}."
                )

        # Mark eliminated players
        for pid in self._active_seat_ids():
            if self.state["chips"][pid] <= 0:
                self.state["eliminated"].add(pid)
                self.state["hand_history"].append(
                    f"  {self._player_name(pid)} is eliminated!"
                )

        # Advance dealer
        active_after = self._active_seat_ids()
        if len(active_after) > 1:
            self.state["dealer_index"] = (
                (self.state["dealer_index"] + 1) % len(active_after)
            )

        self.state["hand_active"] = False
        self.state["needs_new_hand"] = True

    def _player_name(self, pid: str) -> str:
        p = self.players.get(pid)
        return p.info.name if p else pid

    # ------------------------------------------------------------------
    # BaseGame abstract methods
    # ------------------------------------------------------------------

    async def get_next_phase(self) -> GamePhase:
        self.round_number += 1

        # Check if tournament is over
        active_seats = self._active_seat_ids()
        if len(active_seats) <= 1:
            return GamePhase(
                phase_type=PhaseType.GAME_OVER,
                round_number=self.round_number,
                description="Tournament over.",
                active_player_ids=[],
            )

        # Start a new hand if needed
        if self.state["needs_new_hand"]:
            self._start_hand()
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description=(
                    f"Hand #{self.state['hand_number']} begins. "
                    f"Dealer: {self._player_name(self.state['dealer_id'])}. "
                    f"Blinds: {self.state['small_blind']}/{self.state['big_blind']}."
                ),
                active_player_ids=[],
            )

        # If the hand is over, resolve it
        if self._is_hand_over():
            self._resolve_hand()
            summary = "\n".join(self.state["hand_history"][-6:])
            return GamePhase(
                phase_type=PhaseType.RESOLUTION,
                round_number=self.round_number,
                description=f"Hand #{self.state['hand_number']} resolved.\n{summary}",
                active_player_ids=[],
            )

        # Check if betting round is complete and advance street
        if not self.state["action_queue"] or self._is_betting_complete():
            self._advance_street()
            if self.state["street"] == "showdown":
                # Will be resolved on next call
                return await self.get_next_phase()
            if not self.state["action_queue"]:
                # No one can act (all all-in or folded), run out cards
                self._advance_to_showdown()
                return await self.get_next_phase()

        # Next player to act
        if not self.state["action_queue"]:
            # Shouldn't happen but handle gracefully
            self._advance_street()
            return await self.get_next_phase()

        next_pid = self.state["action_queue"].pop(0)

        # Skip players who folded or are all-in
        while next_pid in self.state["folded_players"] or next_pid in self.state["all_in_players"]:
            if not self.state["action_queue"]:
                # No more players to act this round
                return await self.get_next_phase()
            next_pid = self.state["action_queue"].pop(0)

        self.state["acting_player_id"] = next_pid

        street_name = self.state["street"].replace("_", " ").title()
        return GamePhase(
            phase_type=PhaseType.ACTION,
            round_number=self.round_number,
            description=(
                f"Hand #{self.state['hand_number']} - {street_name}: "
                f"{self._player_name(next_pid)}'s turn to act."
            ),
            active_player_ids=[next_pid],
        )

    def _advance_to_showdown(self) -> None:
        """Deal remaining community cards and set street to showdown."""
        deck = self.state["deck"]
        community = self.state["community_cards"]
        while len(community) < 5:
            community.append(deck.pop())
        self.state["community_cards"] = community
        self.state["street"] = "showdown"

    async def get_player_view(self, player_id: str) -> str:
        chips = self.state["chips"]
        player_names = {pid: self._player_name(pid) for pid in self.state["player_ids"]}
        hole_cards = self.state["hole_cards"].get(player_id, [])
        community = self.state["community_cards"]

        return get_turn_prompt(
            player_id=player_id,
            hole_cards=hole_cards,
            community_cards=community,
            pot=self.state["pot"],
            current_bet=self.state["current_bet"],
            player_bet=self.state["round_bets"].get(player_id, 0),
            chips=chips,
            player_names=player_names,
            dealer_id=self.state["dealer_id"],
            street=self.state["street"],
            active_player_ids=self._hand_active_ids(),
            hand_history=self.state["hand_history"],
            hand_number=self.state["hand_number"],
            folded_ids=list(self.state["folded_players"]),
            all_in_ids=list(self.state["all_in_players"]),
            min_raise=self.state.get("min_raise", self.state["big_blind"] * 2),
        )

    async def get_tools_for_player(self, player_id: str) -> list[Callable]:
        return create_poker_tools(self)

    async def process_action(self, player_id: str, action_result: Any) -> ActionResult:
        last = self.state.get("last_action")

        if last is None:
            # The LLM didn't call any tool or something went wrong -- force fold
            self.state["folded_players"].add(player_id)
            self.state["acted_this_round"].add(player_id)
            pname = self._player_name(player_id)
            self.state["hand_history"].append(
                f"  {pname}: fold (no valid action taken, auto-folded)"
            )
            return ActionResult(
                player_id=player_id,
                action_name="fold",
                action_args={},
                result="No valid action was taken. Folded by default.",
                success=True,
            )

        action_name = last["name"]
        action_args = last["args"]
        result_msg = last["result"]
        success = last["success"]

        # Record in hand history
        pname = self._player_name(player_id)
        if success:
            self.state["hand_history"].append(f"  {pname}: {action_name} - {result_msg}")
            self.state["acted_this_round"].add(player_id)

            # If there was a raise, re-open action to other players
            if action_name in ("bet", "raise_bet") and success:
                self._reopen_betting(player_id)
        else:
            # Failed action -- force fold as a fallback if it was an invalid action
            # But only if they keep failing (the framework retries via max_steps)
            self.state["hand_history"].append(
                f"  {pname}: {action_name} (invalid) - {result_msg}"
            )

        # Clear for next action
        self.state["last_action"] = None

        return ActionResult(
            player_id=player_id,
            action_name=action_name,
            action_args=action_args,
            result=result_msg,
            success=success,
            visible_to=None,  # poker actions are public
        )

    def _reopen_betting(self, raiser_id: str) -> None:
        """
        After a bet or raise, add all other active players back to the
        action queue so they get a chance to respond.
        """
        seats = self._active_seat_ids()
        n = len(seats)
        try:
            raiser_idx = seats.index(raiser_id)
        except ValueError:
            return
        # Order: everyone after the raiser, wrapping around, excluding
        # the raiser and anyone folded/all-in
        order: list[str] = []
        for i in range(1, n):
            pid = seats[(raiser_idx + i) % n]
            if (
                pid not in self.state["folded_players"]
                and pid not in self.state["all_in_players"]
            ):
                order.append(pid)
        self.state["action_queue"] = order

    async def check_game_over(self) -> GameOutcome | None:
        active = self._active_seat_ids()
        if len(active) <= 1:
            winner_ids = active
            loser_ids = [
                pid for pid in self.state["player_ids"] if pid not in active
            ]
            # Build ranking by final chip count / elimination order
            ranking = sorted(
                self.state["player_ids"],
                key=lambda pid: self.state["chips"].get(pid, 0),
                reverse=True,
            )
            return GameOutcome(
                game_id=self.game_id,
                game_type="poker",
                winner_ids=winner_ids,
                loser_ids=loser_ids,
                ranking=ranking,
                metadata={
                    "hands_played": self.state["hand_number"],
                    "final_chips": dict(self.state["chips"]),
                },
                timestamp=datetime.now(timezone.utc),
            )

        # Also end if we hit max rounds (hands) as a safety valve
        max_hands = self.config.options.get("max_hands", self.config.max_rounds * 5)
        if self.state["hand_number"] >= max_hands:
            # Player with most chips wins
            ranking = sorted(
                self._active_seat_ids(),
                key=lambda pid: self.state["chips"].get(pid, 0),
                reverse=True,
            )
            winner_ids = [ranking[0]]
            loser_ids = [pid for pid in self.state["player_ids"] if pid != ranking[0]]
            return GameOutcome(
                game_id=self.game_id,
                game_type="poker",
                winner_ids=winner_ids,
                loser_ids=loser_ids,
                ranking=ranking,
                metadata={
                    "hands_played": self.state["hand_number"],
                    "final_chips": dict(self.state["chips"]),
                    "ended_by": "max_hands_reached",
                },
                timestamp=datetime.now(timezone.utc),
            )

        return None
