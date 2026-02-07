from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_arena.games.poker.game import PokerGame


def create_poker_tools(game: PokerGame) -> list[Callable]:
    """
    Return a list of tool functions (closures) that directly mutate
    *game.state* when called.

    After any tool is called, *game.state["last_action"]* is set to a dict
    with keys ``name``, ``args``, ``result``, ``success`` so that
    ``process_action`` can build an ``ActionResult``.
    """

    def _record(name: str, args: dict[str, Any], result: str, success: bool) -> None:
        game.state["last_action"] = {
            "name": name,
            "args": args,
            "result": result,
            "success": success,
        }

    def _acting_player_id() -> str:
        return game.state["acting_player_id"]

    def _player_chips(pid: str) -> int:
        return game.state["chips"][pid]

    def _current_bet() -> int:
        return game.state["current_bet"]

    def _player_round_bet(pid: str) -> int:
        return game.state["round_bets"].get(pid, 0)

    # ----- tool functions -----

    def bet(amount: int) -> str:
        """Place a bet when no bet has been made this round.

        Args:
            amount: The number of chips to bet. Must be at least 1.
        """
        pid = _acting_player_id()
        cur_bet = _current_bet()
        my_round_bet = _player_round_bet(pid)
        my_chips = _player_chips(pid)

        # Can only bet if no one has bet yet (current_bet == 0 or matches your existing blind)
        if cur_bet > my_round_bet:
            _record("bet", {"amount": amount},
                    f"Cannot bet -- there is already a bet of {cur_bet}. Use call or raise_bet.", False)
            return game.state["last_action"]["result"]

        if amount <= 0:
            _record("bet", {"amount": amount}, "Bet amount must be positive.", False)
            return game.state["last_action"]["result"]

        # All-in if they don't have enough
        actual = min(amount, my_chips)

        game.state["chips"][pid] -= actual
        game.state["round_bets"][pid] = my_round_bet + actual
        game.state["pot"] += actual
        game.state["current_bet"] = my_round_bet + actual
        game.state["min_raise"] = (my_round_bet + actual) * 2
        game.state["last_raiser"] = pid

        if game.state["chips"][pid] == 0:
            game.state["all_in_players"].add(pid)

        is_all_in = pid in game.state["all_in_players"]
        msg = f"Bet {actual} chips" + (" (all-in)" if is_all_in else "") + "."
        _record("bet", {"amount": actual}, msg, True)
        return msg

    def call() -> str:
        """Match the current bet to stay in the hand."""
        pid = _acting_player_id()
        cur_bet = _current_bet()
        my_round_bet = _player_round_bet(pid)
        my_chips = _player_chips(pid)
        to_call = cur_bet - my_round_bet

        if to_call <= 0:
            _record("call", {}, "Nothing to call -- use check instead.", False)
            return game.state["last_action"]["result"]

        # All-in if they can't cover the full call
        actual = min(to_call, my_chips)
        game.state["chips"][pid] -= actual
        game.state["round_bets"][pid] = my_round_bet + actual
        game.state["pot"] += actual

        if game.state["chips"][pid] == 0:
            game.state["all_in_players"].add(pid)

        is_all_in = pid in game.state["all_in_players"]
        msg = f"Called {actual} chips" + (" (all-in)" if is_all_in else "") + "."
        _record("call", {}, msg, True)
        return msg

    def raise_bet(total_amount: int) -> str:
        """Raise the current bet to a new total amount.

        Args:
            total_amount: The total bet you want to have in this round. Must be at least double the current bet (or all-in).
        """
        pid = _acting_player_id()
        cur_bet = _current_bet()
        my_round_bet = _player_round_bet(pid)
        my_chips = _player_chips(pid)
        min_raise = game.state.get("min_raise", cur_bet * 2)

        if cur_bet == 0 and my_round_bet == 0:
            _record("raise_bet", {"total_amount": total_amount},
                    "No bet to raise -- use bet instead.", False)
            return game.state["last_action"]["result"]

        # The total_amount is what the player's total round bet should be
        if total_amount <= cur_bet:
            _record("raise_bet", {"total_amount": total_amount},
                    f"Raise must be more than the current bet of {cur_bet}.", False)
            return game.state["last_action"]["result"]

        additional_needed = total_amount - my_round_bet
        if additional_needed <= 0:
            _record("raise_bet", {"total_amount": total_amount},
                    f"You have already put in {my_round_bet}. Raise total must exceed that.", False)
            return game.state["last_action"]["result"]

        # Check minimum raise (but allow all-in for less)
        if total_amount < min_raise and additional_needed < my_chips:
            _record("raise_bet", {"total_amount": total_amount},
                    f"Minimum raise is to {min_raise}. Raise more or go all-in.", False)
            return game.state["last_action"]["result"]

        actual = min(additional_needed, my_chips)
        new_total = my_round_bet + actual

        game.state["chips"][pid] -= actual
        game.state["round_bets"][pid] = new_total
        game.state["pot"] += actual
        # The raise amount is the difference between new bet and old bet
        raise_increment = new_total - cur_bet
        game.state["current_bet"] = new_total
        game.state["min_raise"] = new_total + max(raise_increment, 1)
        game.state["last_raiser"] = pid

        if game.state["chips"][pid] == 0:
            game.state["all_in_players"].add(pid)

        is_all_in = pid in game.state["all_in_players"]
        msg = f"Raised to {new_total} chips total" + (" (all-in)" if is_all_in else "") + "."
        _record("raise_bet", {"total_amount": new_total}, msg, True)
        return msg

    def fold() -> str:
        """Fold your hand and forfeit any chips already in the pot."""
        pid = _acting_player_id()
        game.state["folded_players"].add(pid)
        msg = "Folded."
        _record("fold", {}, msg, True)
        return msg

    def check() -> str:
        """Pass your action without betting (only valid if no bet to match)."""
        pid = _acting_player_id()
        cur_bet = _current_bet()
        my_round_bet = _player_round_bet(pid)

        if cur_bet > my_round_bet:
            _record("check", {},
                    f"Cannot check -- there is a bet of {cur_bet} to match. Call, raise, or fold.",
                    False)
            return game.state["last_action"]["result"]

        msg = "Checked."
        _record("check", {}, msg, True)
        return msg

    return [bet, call, raise_bet, fold, check]
