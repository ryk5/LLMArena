from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from llm_arena.games.impostor.roles import LOCATIONS

if TYPE_CHECKING:
    from llm_arena.games.impostor.game import ImpostorGame


def create_impostor_tools(game: ImpostorGame) -> dict[str, list[Callable]]:
    """
    Factory that creates tool functions for the Impostor game.

    Tools are closures that capture the game instance so they can read and
    mutate game state directly. This follows the pattern used by the Dedalus
    SDK: plain functions with type hints and docstrings.

    Returns a dict keyed by phase name -> list of tool callables:
      - "action": movement, task, kill, report, meeting tools
      - "discussion": make_statement tool
      - "voting": cast_vote, skip_vote tools
    """

    # -----------------------------------------------------------------------
    # ACTION phase tools
    # -----------------------------------------------------------------------

    def move_to(location: str) -> str:
        """Move to a different location on the space station.

        Args:
            location: The name of the location to move to. Must be one of:
                      Cafeteria, Reactor, Electrical, MedBay, Navigation,
                      Security, Admin, Storage.
        """
        player_id = game.state["current_player_id"]
        player_name = game.players[player_id].info.name

        # Normalize the location name (capitalize each word)
        location_normalized = location.strip().title()

        # Handle common variations
        location_map = {loc.lower(): loc for loc in LOCATIONS}
        lookup = location_normalized.lower()
        if lookup in location_map:
            location_normalized = location_map[lookup]
        elif location_normalized not in LOCATIONS:
            result = (
                f"Invalid location: '{location}'. "
                f"Valid locations: {', '.join(LOCATIONS)}"
            )
            game.state["last_action"] = {
                "tool": "move_to",
                "args": {"location": location},
                "success": False,
                "result": result,
            }
            return result

        old_location = game.state["locations"][player_id]
        game.state["locations"][player_id] = location_normalized

        # See who else is at the new location
        others_here = [
            game.players[pid].info.name
            for pid in game.state["alive"]
            if pid != player_id
            and game.state["locations"][pid] == location_normalized
            and game.state["alive"][pid]
        ]

        # Check for dead bodies at new location
        bodies_here = [
            game.players[pid].info.name
            for pid in game.state["dead_bodies"]
            if game.state["dead_bodies"][pid] == location_normalized
        ]

        parts = [f"You moved from {old_location} to {location_normalized}."]
        if others_here:
            parts.append(f"Players here: {', '.join(others_here)}.")
        else:
            parts.append("No other players here.")
        if bodies_here:
            parts.append(f"WARNING: You see the body of {', '.join(bodies_here)} here!")

        result = " ".join(parts)
        game.state["last_action"] = {
            "tool": "move_to",
            "args": {"location": location_normalized},
            "success": True,
            "result": result,
        }
        return result

    def do_task() -> str:
        """Complete a task at your current location.

        You must be at the correct location for one of your assigned tasks.
        Impostors can call this to fake doing a task, but it will not count
        toward task progress.
        """
        player_id = game.state["current_player_id"]
        player_name = game.players[player_id].info.name
        role = game.roles[player_id]
        location = game.state["locations"][player_id]

        if role.team == "impostor":
            # Impostors fake tasks -- looks normal to others
            result = f"You pretended to do a task at {location}. (No actual progress.)"
            game.state["last_action"] = {
                "tool": "do_task",
                "args": {},
                "success": True,
                "result": result,
            }
            return result

        # Find an incomplete task at this location
        tasks = game.state["tasks"].get(player_id, [])
        for task in tasks:
            if task["location"] == location and not task["completed"]:
                task["completed"] = True
                game.state["completed_tasks"] += 1

                remaining = sum(1 for t in tasks if not t["completed"])
                total = game.state["total_tasks"]
                completed = game.state["completed_tasks"]
                pct = (completed / total * 100) if total > 0 else 0

                result = (
                    f"Task completed: '{task['task']}' at {location}. "
                    f"You have {remaining} task(s) remaining. "
                    f"Overall progress: {completed}/{total} ({pct:.0f}%)."
                )
                game.state["last_action"] = {
                    "tool": "do_task",
                    "args": {},
                    "success": True,
                    "result": result,
                }
                return result

        # No task available here
        available = [
            f"'{t['task']}' at {t['location']}"
            for t in tasks
            if not t["completed"]
        ]
        if available:
            result = (
                f"You have no task to do at {location}. "
                f"Your remaining tasks: {'; '.join(available)}."
            )
        else:
            result = "All your tasks are already completed!"

        game.state["last_action"] = {
            "tool": "do_task",
            "args": {},
            "success": False,
            "result": result,
        }
        return result

    def kill_player(player_name: str) -> str:
        """Kill a crewmate at your current location. Impostor only.

        Args:
            player_name: The name of the player to kill. They must be alive
                         and at the same location as you.
        """
        player_id = game.state["current_player_id"]
        killer_name = game.players[player_id].info.name
        role = game.roles[player_id]
        location = game.state["locations"][player_id]

        if role.team != "impostor":
            result = "Only impostors can kill. You are a crewmate."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Check cooldown
        cooldown = game.state["kill_cooldowns"].get(player_id, 0)
        if cooldown > 0:
            result = f"Kill is on cooldown for {cooldown} more round(s)."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Find the target player by name
        target_id = None
        for pid, player in game.players.items():
            if player.info.name.lower() == player_name.strip().lower():
                target_id = pid
                break

        if target_id is None:
            result = f"No player named '{player_name}' found."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Cannot kill yourself
        if target_id == player_id:
            result = "You cannot kill yourself."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Target must be alive
        if not game.state["alive"].get(target_id, False):
            result = f"{player_name} is already dead."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Cannot kill fellow impostors
        if game.roles[target_id].team == "impostor":
            result = f"{player_name} is a fellow impostor. You cannot kill them."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Target must be at same location
        target_location = game.state["locations"][target_id]
        if target_location != location:
            result = f"{player_name} is not at your location ({location})."
            game.state["last_action"] = {
                "tool": "kill_player",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        # Execute the kill
        game.state["alive"][target_id] = False
        game.state["dead_bodies"][target_id] = location
        game.state["kill_cooldowns"][player_id] = 2  # 2-round cooldown

        # Track for discussion context
        if "recently_killed" not in game.state:
            game.state["recently_killed"] = []
        game.state["recently_killed"].append(game.players[target_id].info.name)

        result = (
            f"You killed {game.players[target_id].info.name} at {location}. "
            f"Their body remains here. Kill cooldown: 2 rounds."
        )
        game.state["last_action"] = {
            "tool": "kill_player",
            "args": {"player_name": player_name},
            "success": True,
            "result": result,
            "victim_id": target_id,
        }
        return result

    def report_body() -> str:
        """Report a dead body at your current location.

        This will trigger a discussion and voting phase. You can only report
        if there is an unreported body at your current location.
        """
        player_id = game.state["current_player_id"]
        player_name = game.players[player_id].info.name
        location = game.state["locations"][player_id]

        # Find bodies at this location
        bodies_here = [
            pid for pid in game.state["dead_bodies"]
            if game.state["dead_bodies"][pid] == location
        ]

        if not bodies_here:
            result = f"There are no dead bodies at {location} to report."
            game.state["last_action"] = {
                "tool": "report_body",
                "args": {},
                "success": False,
                "result": result,
            }
            return result

        # Report the first body found
        body_id = bodies_here[0]
        body_name = game.players[body_id].info.name

        # Set the meeting trigger
        game.state["meeting_trigger"] = "body_report"
        game.state["meeting_caller"] = player_name
        game.state["reported_body"] = body_name
        game.state["meeting_triggered"] = True

        # Remove reported bodies from the dead_bodies tracker
        for bid in bodies_here:
            del game.state["dead_bodies"][bid]

        result = (
            f"You reported the body of {body_name} at {location}! "
            f"An emergency discussion will now begin."
        )
        game.state["last_action"] = {
            "tool": "report_body",
            "args": {},
            "success": True,
            "result": result,
        }
        return result

    def call_emergency_meeting() -> str:
        """Call an emergency meeting. Each player can only do this once per game.

        This triggers a discussion and voting phase immediately.
        """
        player_id = game.state["current_player_id"]
        player_name = game.players[player_id].info.name

        meetings_left = game.state["emergency_meetings"].get(player_id, 0)
        if meetings_left <= 0:
            result = "You have no emergency meetings remaining."
            game.state["last_action"] = {
                "tool": "call_emergency_meeting",
                "args": {},
                "success": False,
                "result": result,
            }
            return result

        game.state["emergency_meetings"][player_id] -= 1
        game.state["meeting_trigger"] = "emergency_meeting"
        game.state["meeting_caller"] = player_name
        game.state["meeting_triggered"] = True

        result = (
            f"You called an emergency meeting! "
            f"All players will now discuss and vote."
        )
        game.state["last_action"] = {
            "tool": "call_emergency_meeting",
            "args": {},
            "success": True,
            "result": result,
        }
        return result

    # -----------------------------------------------------------------------
    # DISCUSSION phase tools
    # -----------------------------------------------------------------------

    def make_statement(statement: str) -> str:
        """Make a statement during the discussion phase.

        Args:
            statement: What you want to say to the group. Share observations,
                       make accusations, defend yourself, or ask questions.
        """
        player_id = game.state["current_player_id"]
        player_name = game.players[player_id].info.name

        game.state["last_action"] = {
            "tool": "make_statement",
            "args": {"statement": statement},
            "success": True,
            "result": statement,
        }
        return f"You said: {statement}"

    # -----------------------------------------------------------------------
    # VOTING phase tools
    # -----------------------------------------------------------------------

    def cast_vote(player_name: str) -> str:
        """Vote to eject a player from the space station.

        Args:
            player_name: The name of the player you want to eject.
        """
        voter_id = game.state["current_player_id"]
        voter_name = game.players[voter_id].info.name

        # Find target by name
        target_id = None
        for pid, player in game.players.items():
            if player.info.name.lower() == player_name.strip().lower():
                target_id = pid
                break

        if target_id is None:
            result = f"No player named '{player_name}' found. Vote not counted."
            game.state["last_action"] = {
                "tool": "cast_vote",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        if not game.state["alive"].get(target_id, False):
            result = f"{player_name} is dead and cannot be voted for."
            game.state["last_action"] = {
                "tool": "cast_vote",
                "args": {"player_name": player_name},
                "success": False,
                "result": result,
            }
            return result

        if target_id == voter_id:
            # Allow self-voting (weird but valid strategy)
            pass

        # Record the vote
        if "votes" not in game.state:
            game.state["votes"] = {}
        game.state["votes"][voter_id] = target_id

        target_display = game.players[target_id].info.name
        result = f"You voted to eject {target_display}."
        game.state["last_action"] = {
            "tool": "cast_vote",
            "args": {"player_name": player_name},
            "success": True,
            "result": result,
        }
        return result

    def skip_vote() -> str:
        """Skip your vote. You choose not to eject anyone this round."""
        voter_id = game.state["current_player_id"]

        if "votes" not in game.state:
            game.state["votes"] = {}
        game.state["votes"][voter_id] = "skip"

        result = "You chose to skip your vote."
        game.state["last_action"] = {
            "tool": "skip_vote",
            "args": {},
            "success": True,
            "result": result,
        }
        return result

    # -----------------------------------------------------------------------
    # Return tools grouped by phase
    # -----------------------------------------------------------------------

    return {
        "action_crew": [move_to, do_task, report_body, call_emergency_meeting],
        "action_impostor": [move_to, do_task, kill_player, report_body, call_emergency_meeting],
        "discussion": [make_statement],
        "voting": [cast_vote, skip_vote],
    }
