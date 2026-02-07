from __future__ import annotations

from itertools import combinations

# ---------------------------------------------------------------------------
# Card representation helpers
# ---------------------------------------------------------------------------

RANK_CHARS = "23456789TJQKA"
RANK_MAP: dict[str, int] = {c: i for i, c in enumerate(RANK_CHARS)}  # 0..12
SUIT_CHARS = "cdhs"


def card_rank(card: str) -> int:
    """Return numeric rank 0-12 for a card like 'Ah' or 'Td'."""
    return RANK_MAP[card[0]]


def card_suit(card: str) -> str:
    """Return the suit character for a card like 'Ah'."""
    return card[1]


# ---------------------------------------------------------------------------
# Hand ranking constants  (higher is better)
# ---------------------------------------------------------------------------

HAND_HIGH_CARD = 0
HAND_ONE_PAIR = 1
HAND_TWO_PAIR = 2
HAND_THREE_OF_A_KIND = 3
HAND_STRAIGHT = 4
HAND_FLUSH = 5
HAND_FULL_HOUSE = 6
HAND_FOUR_OF_A_KIND = 7
HAND_STRAIGHT_FLUSH = 8
HAND_ROYAL_FLUSH = 9

HAND_NAMES = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
    "Royal Flush",
]


def hand_rank_name(rank: int) -> str:
    """Human-readable name for a hand rank constant."""
    if 0 <= rank < len(HAND_NAMES):
        return HAND_NAMES[rank]
    return "Unknown"


# ---------------------------------------------------------------------------
# 5-card evaluation
# ---------------------------------------------------------------------------

def _evaluate_five(cards: list[str]) -> tuple[int, list[int]]:
    """
    Evaluate exactly 5 cards and return (rank, tiebreakers).

    *rank* is one of the HAND_* constants (0-9).
    *tiebreakers* is a list of ints used to break ties, compared
    lexicographically (higher is better).
    """
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]

    is_flush = len(set(suits)) == 1

    # Check for straight (including A-2-3-4-5 wheel)
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Wheel: A-2-3-4-5 (ranks 12,3,2,1,0)
        elif unique_ranks == [12, 3, 2, 1, 0]:
            is_straight = True
            straight_high = 3  # 5-high straight

    # Count rank frequencies
    from collections import Counter
    freq = Counter(ranks)
    # Group by frequency then by rank (descending) for ordering
    # e.g. [(3, 10), (2, 5)] means three tens and two fives
    groups = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    freq_pattern = tuple(g[1] for g in groups)

    if is_straight and is_flush:
        if straight_high == 12:  # Ace-high straight flush
            return (HAND_ROYAL_FLUSH, [straight_high])
        return (HAND_STRAIGHT_FLUSH, [straight_high])

    if freq_pattern == (4, 1):
        quad_rank = groups[0][0]
        kicker = groups[1][0]
        return (HAND_FOUR_OF_A_KIND, [quad_rank, kicker])

    if freq_pattern == (3, 2):
        trip_rank = groups[0][0]
        pair_rank = groups[1][0]
        return (HAND_FULL_HOUSE, [trip_rank, pair_rank])

    if is_flush:
        return (HAND_FLUSH, ranks)

    if is_straight:
        return (HAND_STRAIGHT, [straight_high])

    if freq_pattern == (3, 1, 1):
        trip_rank = groups[0][0]
        kickers = sorted([groups[1][0], groups[2][0]], reverse=True)
        return (HAND_THREE_OF_A_KIND, [trip_rank] + kickers)

    if freq_pattern == (2, 2, 1):
        high_pair = max(groups[0][0], groups[1][0])
        low_pair = min(groups[0][0], groups[1][0])
        kicker = groups[2][0]
        return (HAND_TWO_PAIR, [high_pair, low_pair, kicker])

    if freq_pattern == (2, 1, 1, 1):
        pair_rank = groups[0][0]
        kickers = sorted([groups[1][0], groups[2][0], groups[3][0]], reverse=True)
        return (HAND_ONE_PAIR, [pair_rank] + kickers)

    # High card
    return (HAND_HIGH_CARD, ranks)


# ---------------------------------------------------------------------------
# 7-card (Texas Hold'em) evaluation
# ---------------------------------------------------------------------------

def evaluate_hand(
    hole_cards: list[str], community_cards: list[str]
) -> tuple[int, list[int]]:
    """
    Evaluate the best 5-card hand from *hole_cards* (2) and
    *community_cards* (3-5).

    Returns (rank, tiebreaker_values) where *rank* is a HAND_* constant
    and *tiebreaker_values* is a list of ints for lexicographic comparison.
    """
    all_cards = hole_cards + community_cards
    if len(all_cards) < 5:
        raise ValueError(
            f"Need at least 5 cards to evaluate, got {len(all_cards)}"
        )

    best: tuple[int, list[int]] | None = None
    for combo in combinations(all_cards, 5):
        result = _evaluate_five(list(combo))
        if best is None or _compare_single(result, best) > 0:
            best = result

    assert best is not None
    return best


def _compare_single(
    a: tuple[int, list[int]], b: tuple[int, list[int]]
) -> int:
    """Return >0 if *a* beats *b*, <0 if *b* beats *a*, 0 for tie."""
    if a[0] != b[0]:
        return a[0] - b[0]
    for av, bv in zip(a[1], b[1]):
        if av != bv:
            return av - bv
    return 0


def compare_hands(
    hands: list[tuple[int, list[int]]],
) -> list[int]:
    """
    Given a list of evaluated hands (each from `evaluate_hand`), return
    the indices of the winner(s).  Ties are possible.
    """
    if not hands:
        return []

    best_idx = [0]
    for i in range(1, len(hands)):
        cmp = _compare_single(hands[i], hands[best_idx[0]])
        if cmp > 0:
            best_idx = [i]
        elif cmp == 0:
            best_idx.append(i)
    return best_idx


def rank_name_for_cards(
    hole_cards: list[str], community_cards: list[str]
) -> str:
    """Convenience: return the human-readable hand name."""
    rank, _ = evaluate_hand(hole_cards, community_cards)
    return hand_rank_name(rank)
