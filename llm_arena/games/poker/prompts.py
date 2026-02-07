from __future__ import annotations

from typing import Any


POKER_SYSTEM_PROMPT = """\
You are an AI playing Texas Hold'em poker in a competitive tournament against \
other AI models.

## Rules of Texas Hold'em

- Each player is dealt 2 private "hole" cards.
- Five community cards are dealt face-up over several rounds:
  - **Flop**: 3 cards
  - **Turn**: 1 card
  - **River**: 1 card
- You make the best 5-card hand from your 2 hole cards + 5 community cards.
- Hand rankings (best to worst):
  Royal Flush > Straight Flush > Four of a Kind > Full House > Flush > \
Straight > Three of a Kind > Two Pair > One Pair > High Card

## Betting

- Each round, players can **check**, **bet**, **call**, **raise**, or **fold**.
- **Check**: pass your action (only if no bet has been made this round).
- **Bet**: place chips into the pot (only if no bet has been made this round). \
Use the `bet` tool with the amount you want to bet.
- **Call**: match the current bet.
- **Raise**: increase the current bet. Use the `raise_bet` tool with the TOTAL \
amount you want your bet to be (not the additional amount). Your raise must be \
at least double the current bet.
- **Fold**: surrender your hand and forfeit any chips in the pot.
- Going **all-in**: you can always bet or call with all your remaining chips, \
even if you don't have enough to match a bet/raise.

## Blinds

- The player to the left of the dealer posts the **small blind** (half the \
minimum bet).
- The next player posts the **big blind** (the minimum bet).
- Pre-flop, the player to the left of the big blind acts first.
- Post-flop, the first active player to the left of the dealer acts first.

## Strategy Tips

- Consider your position relative to the dealer.
- Think about pot odds and implied odds.
- Pay attention to betting patterns of opponents.
- Don't be afraid to fold weak hands.
- Bluffing can work, but use it judiciously.
- Manage your chip stack -- don't go broke on marginal hands.

Play your best poker. Good luck!
"""


def get_turn_prompt(
    player_id: str,
    hole_cards: list[str],
    community_cards: list[str],
    pot: int,
    current_bet: int,
    player_bet: int,
    chips: dict[str, int],
    player_names: dict[str, str],
    dealer_id: str,
    street: str,
    active_player_ids: list[str],
    hand_history: list[str],
    hand_number: int,
    folded_ids: list[str],
    all_in_ids: list[str],
    min_raise: int,
) -> str:
    """Build the per-turn prompt showing the current game state to a player."""
    name = player_names.get(player_id, player_id)
    hole_str = " ".join(hole_cards) if hole_cards else "(not yet dealt)"
    community_str = " ".join(community_cards) if community_cards else "(none)"
    street_display = street.replace("_", " ").title()

    to_call = max(0, current_bet - player_bet)
    my_chips = chips.get(player_id, 0)

    # Build the chip stacks / status table
    stack_lines: list[str] = []
    for pid in sorted(chips.keys()):
        pname = player_names.get(pid, pid)
        marker_parts: list[str] = []
        if pid == dealer_id:
            marker_parts.append("DEALER")
        if pid in folded_ids:
            marker_parts.append("FOLDED")
        if pid in all_in_ids:
            marker_parts.append("ALL-IN")
        if pid == player_id:
            marker_parts.append("YOU")
        marker = f" ({', '.join(marker_parts)})" if marker_parts else ""
        stack_lines.append(f"  {pname}: {chips[pid]} chips{marker}")

    stacks_str = "\n".join(stack_lines)

    # Available actions hint
    action_hints: list[str] = []
    if current_bet == 0 or current_bet == player_bet:
        action_hints.append("check (no cost)")
        action_hints.append(f"bet <amount> (place a new bet, minimum {max(1, current_bet or 1)})")
    if current_bet > player_bet:
        action_hints.append(f"call (costs {to_call} chips)")
    if current_bet > 0:
        action_hints.append(
            f"raise_bet <total> (raise to at least {min_raise})"
        )
    action_hints.append("fold (surrender hand)")

    actions_str = "\n".join(f"  - {h}" for h in action_hints)

    # Hand history
    history_str = "\n".join(hand_history) if hand_history else "(no actions yet)"

    prompt = f"""\
## Hand #{hand_number} -- {street_display}

### Your Hole Cards
  {hole_str}

### Community Cards
  {community_str}

### Pot
  {pot} chips

### Current Bet to Match
  {current_bet} chips (you have put in {player_bet} this round, {to_call} more to call)

### Your Chips
  {my_chips}

### All Players
{stacks_str}

### Available Actions
{actions_str}

### Hand History
{history_str}

---
It is your turn, {name}. Choose one action.
"""
    return prompt
