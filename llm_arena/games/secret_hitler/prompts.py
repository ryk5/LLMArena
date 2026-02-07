from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_arena.games.secret_hitler.game import SecretHitlerGame

from llm_arena.games.secret_hitler.roles import (
    get_fascist_team,
    get_hitler,
    get_regular_fascists,
    hitler_knows_fascists,
)

# ---------------------------------------------------------------------------
# System prompt (sent as `instructions` to each LLM player)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are playing Secret Hitler, a social deduction board game.

## Rules Summary
- Players are secretly assigned roles: Liberal, Fascist, or Hitler.
- There is a policy deck containing Liberal and Fascist policy tiles.
- Each round, a President nominates a Chancellor. All players vote Ja (yes) or \
Nein (no). If the vote passes, the President draws 3 policies, discards 1, and \
passes 2 to the Chancellor, who enacts 1.
- Liberals win by enacting 5 Liberal policies OR assassinating Hitler.
- Fascists win by enacting 6 Fascist policies OR electing Hitler as Chancellor \
after 3+ Fascist policies are already enacted.
- If a vote fails, the election tracker advances. After 3 consecutive failed \
elections, the top policy on the deck is automatically enacted.
- The Presidency rotates clockwise each round.
- Term limits: the previous President and Chancellor cannot be nominated as \
Chancellor for the current round (unless there are only 5 players alive, in \
which case only the previous Chancellor is term-limited).

## Presidential Powers
After certain Fascist policies are enacted, the President receives a special power:
- In a 7-8 player game: 2nd Fascist policy = Investigate Loyalty; \
3rd = Special Election; 4th = Execution; 5th = Execution.
- In a 9-10 player game: 1st Fascist policy = Investigate Loyalty; \
2nd = Investigate Loyalty; 3rd = Special Election; 4th = Execution; \
5th = Execution.
- In a 5-6 player game: 3rd Fascist policy = Peek at top 3 policies; \
4th = Execution; 5th = Execution.

## Strategy Tips
- As a Liberal: watch for inconsistencies, ask questions, vote carefully.
- As a Fascist: deflect suspicion, support Hitler covertly, enact Fascist policies \
when possible while maintaining cover.
- As Hitler: act like a Liberal, build trust, avoid suspicion at all costs.

Use the tools provided for each phase to take your action. Only call one tool per \
turn unless instructed otherwise.
"""


# ---------------------------------------------------------------------------
# Helper to build role-knowledge string
# ---------------------------------------------------------------------------

def _role_knowledge(game: SecretHitlerGame, player_id: str) -> str:
    """Build the secret knowledge paragraph a player should see."""
    role = game.roles[player_id]
    player_name = game.players[player_id].info.name
    lines: list[str] = []
    lines.append(f"Your name is {player_name}.")
    lines.append(f"Your secret role is: {role.name} ({role.team} team).")

    if role.name == "Fascist":
        # Regular fascists always know the full team
        fascist_team = get_fascist_team(game.roles)
        hitler_id = get_hitler(game.roles)
        teammates = []
        for fid in fascist_team:
            fname = game.players[fid].info.name
            frole = game.roles[fid].name
            if fid != player_id:
                teammates.append(f"{fname} ({frole})")
        lines.append(
            "Your Fascist teammates are: " + ", ".join(teammates) + "."
        )
    elif role.name == "Hitler":
        num_players = len(game.players)
        if hitler_knows_fascists(num_players):
            fascists = get_regular_fascists(game.roles)
            names = [game.players[fid].info.name for fid in fascists]
            lines.append(
                "Since this is a small game (5-6 players), you know your "
                "fellow Fascist(s): " + ", ".join(names) + "."
            )
        else:
            lines.append(
                "In games with 7+ players, you do NOT know who the other "
                "Fascists are. You must figure it out from their behaviour."
            )
    else:
        # Liberal
        lines.append("You do not know anyone's secret role.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Board state summary (visible to all)
# ---------------------------------------------------------------------------

def _board_state(game: SecretHitlerGame) -> str:
    """Build a public board-state summary."""
    state = game.state
    lines: list[str] = [
        "## Current Board State",
        f"- Liberal policies enacted: {state['liberal_policies']} / 5",
        f"- Fascist policies enacted: {state['fascist_policies']} / 6",
        f"- Election tracker (failed elections): {state['election_tracker']} / 3",
        f"- Round number: {game.round_number}",
    ]

    # Alive players
    alive = _alive_player_names(game)
    lines.append(f"- Players alive: {', '.join(alive)}")

    # Dead players
    dead = _dead_player_names(game)
    if dead:
        lines.append(f"- Players eliminated: {', '.join(dead)}")

    # Term-limited players
    term_limited = state.get("term_limited_ids", [])
    if term_limited:
        tl_names = [game.players[pid].info.name for pid in term_limited]
        lines.append(
            f"- Cannot be nominated as Chancellor this round (term-limited): "
            f"{', '.join(tl_names)}"
        )

    # Current president
    president_id = state.get("president_id")
    if president_id:
        lines.append(
            f"- Current Presidential Candidate: "
            f"{game.players[president_id].info.name}"
        )

    # Previous government
    prev_pres = state.get("prev_president_id")
    prev_chan = state.get("prev_chancellor_id")
    if prev_pres:
        lines.append(
            f"- Previous President: {game.players[prev_pres].info.name}"
        )
    if prev_chan:
        lines.append(
            f"- Previous Chancellor: {game.players[prev_chan].info.name}"
        )

    # Recently enacted policies log
    policy_log = state.get("policy_log", [])
    if policy_log:
        recent = policy_log[-5:]
        log_strs = [f"Round {p['round']}: {p['policy']} (P: {p['president']}, C: {p['chancellor']})"
                     for p in recent]
        lines.append("- Recent policy enactments: " + "; ".join(log_strs))

    return "\n".join(lines)


def _alive_player_names(game: SecretHitlerGame) -> list[str]:
    alive_ids = game.state["alive_ids"]
    return [game.players[pid].info.name for pid in alive_ids]


def _dead_player_names(game: SecretHitlerGame) -> list[str]:
    all_ids = set(game.players.keys())
    alive_ids = set(game.state["alive_ids"])
    dead_ids = all_ids - alive_ids
    return [game.players[pid].info.name for pid in dead_ids]


# ---------------------------------------------------------------------------
# Phase-specific prompt builders
# ---------------------------------------------------------------------------

def discussion_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the discussion phase at the start of each round."""
    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Phase: Discussion",
        f"Round {game.round_number} discussion. The current Presidential "
        f"Candidate is {game.players[game.state['president_id']].info.name}.",
        "",
        "Share your thoughts with the group. You might discuss who should be "
        "nominated as Chancellor, whether you trust the current President, "
        "or any suspicions you have. Use the `make_statement` tool.",
    ]
    return "\n".join(parts)


def nomination_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the President to nominate a Chancellor."""
    eligible = game.state.get("eligible_chancellor_ids", [])
    eligible_names = [game.players[pid].info.name for pid in eligible]

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Phase: Chancellor Nomination",
        f"You are the President ({game.players[player_id].info.name}). "
        f"Nominate a Chancellor from the eligible players.",
        f"Eligible players: {', '.join(eligible_names)}",
        "",
        "Use the `nominate_chancellor` tool with the name of the player you "
        "want to nominate.",
    ]
    return "\n".join(parts)


def voting_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for all alive players to vote on the proposed government."""
    president_name = game.players[game.state["president_id"]].info.name
    chancellor_name = game.players[game.state["chancellor_nominee_id"]].info.name

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Phase: Vote on Government",
        f"The proposed government is:",
        f"  - President: {president_name}",
        f"  - Chancellor: {chancellor_name}",
        "",
        "Vote 'ja' (yes) to approve this government, or 'nein' (no) to reject it.",
        "Use the `cast_vote` tool.",
    ]

    # Remind fascists about the Hitler-chancellor win condition
    if game.state["fascist_policies"] >= 3:
        parts.append(
            "\n**IMPORTANT**: 3 or more Fascist policies have been enacted. "
            "If Hitler is elected Chancellor, the Fascists win immediately!"
        )

    return "\n".join(parts)


def president_discard_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the President to discard one of 3 drawn policies."""
    drawn = game.state["drawn_policies"]
    policy_display = ", ".join(
        f"[{i}] {p}" for i, p in enumerate(drawn)
    )

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Phase: Presidential Legislation",
        "You have drawn 3 policy tiles from the deck. You must discard one "
        "and pass the remaining 2 to the Chancellor.",
        f"Your drawn policies: {policy_display}",
        "",
        "Use the `discard_policy` tool with the index (0, 1, or 2) of the "
        "policy you want to DISCARD.",
        "",
        "Remember: you may say whatever you want about the cards you drew, "
        "but you are not required to tell the truth.",
    ]
    return "\n".join(parts)


def chancellor_enact_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the Chancellor to enact one of 2 remaining policies."""
    remaining = game.state["chancellor_policies"]
    policy_display = ", ".join(
        f"[{i}] {p}" for i, p in enumerate(remaining)
    )

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Phase: Chancellor Legislation",
        "The President has passed you 2 policy tiles. You must enact one.",
        f"Your policy choices: {policy_display}",
        "",
        "Use the `enact_policy` tool with the index (0 or 1) of the policy "
        "you want to ENACT.",
    ]
    return "\n".join(parts)


def investigation_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the President to investigate a player's loyalty."""
    alive_ids = game.state["alive_ids"]
    investigable = [pid for pid in alive_ids if pid != player_id]
    names = [game.players[pid].info.name for pid in investigable]

    # Exclude already-investigated players (optional flavour)
    already = game.state.get("investigated_ids", [])
    if already:
        already_names = [game.players[pid].info.name for pid in already]
        names_note = (
            f"\nPlayers you have already investigated: "
            f"{', '.join(already_names)}. You may investigate them again, "
            f"but it is usually more useful to investigate someone new."
        )
    else:
        names_note = ""

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Presidential Power: Investigate Loyalty",
        "You may investigate one player's party membership card. You will learn "
        "whether they are a Liberal or a Fascist (note: Hitler's membership card "
        "says 'Fascist').",
        f"Eligible targets: {', '.join(names)}" + names_note,
        "",
        "Use the `investigate_player` tool with the player's name.",
    ]
    return "\n".join(parts)


def execution_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the President to execute a player."""
    alive_ids = game.state["alive_ids"]
    targets = [pid for pid in alive_ids if pid != player_id]
    names = [game.players[pid].info.name for pid in targets]

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Presidential Power: Execution",
        "You must choose a player to execute. That player is immediately "
        "eliminated from the game. If you execute Hitler, the Liberals win!",
        f"Eligible targets: {', '.join(names)}",
        "",
        "Use the `execute_player` tool with the player's name.",
    ]
    return "\n".join(parts)


def special_election_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt for the President to choose the next Presidential Candidate."""
    alive_ids = game.state["alive_ids"]
    targets = [pid for pid in alive_ids if pid != player_id]
    names = [game.players[pid].info.name for pid in targets]

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Presidential Power: Special Election",
        "You may choose any other living player to be the next Presidential "
        "Candidate. After their turn, the presidency returns to the normal "
        "rotation order.",
        f"Eligible choices: {', '.join(names)}",
        "",
        "Use the `choose_next_president` tool with the player's name.",
    ]
    return "\n".join(parts)


def peek_prompt(game: SecretHitlerGame, player_id: str) -> str:
    """Prompt shown when the President peeks at top 3 policies (5-6 player power)."""
    top_three = game.state.get("peeked_policies", [])
    display = ", ".join(top_three)

    parts = [
        SYSTEM_PROMPT,
        "",
        _role_knowledge(game, player_id),
        "",
        _board_state(game),
        "",
        "## Presidential Power: Policy Peek",
        f"You peek at the top 3 policies on the deck: {display}",
        "",
        "This information is for your eyes only. You may share it (or lie "
        "about it) during the next discussion phase.",
        "Use the `make_statement` tool to acknowledge you have seen the policies.",
    ]
    return "\n".join(parts)


def vote_result_summary(
    game: SecretHitlerGame,
    votes: dict[str, str],
    passed: bool,
) -> str:
    """Build a summary of the vote results for logging / display."""
    ja_names = [game.players[pid].info.name for pid, v in votes.items() if v == "ja"]
    nein_names = [game.players[pid].info.name for pid, v in votes.items() if v == "nein"]
    result = "PASSED" if passed else "FAILED"
    return (
        f"Vote {result}. Ja ({len(ja_names)}): {', '.join(ja_names) or 'none'}. "
        f"Nein ({len(nein_names)}): {', '.join(nein_names) or 'none'}."
    )
