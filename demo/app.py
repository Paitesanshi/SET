# app.py

import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict, field
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from agentscope.message import Msg
import sys

# If needed, adjust your system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from set import (
    TradeProposal, RoundTradeState, ResourceAgent, UserAgent, 
    SERoom, SimulationAnalyzer, compute_value,
    update_bdi_with_llm, update_affinity_with_llm
)

import agentscope

# =============================
# Global / Static Config
# =============================

app = Flask(__name__)
app.secret_key = "YOUR_FLASK_SECRET_KEY"  
CORS(app)

# Simulation config (shared across sessions)
SIMULATION_CONFIG = {
    "resources": {
        "Resource A": {"production": 15},
        "Resource B": {"production": 15},
        "Resource C": {"production": 15}
    },
    "initial_resources": {
        "Resource A": 5,
        "Resource B": 5,
        "Resource C": 5
    },
    "resource_values": {
        "Resource A": 1,
        "Resource B": 1,
        "Resource C": 1,
        "Resource A + Resource B": 4,
        "Resource A + Resource C": 4,
        "Resource B + Resource C": 4,
        "Resource A + Resource B + Resource C": 9
    },
    "svo": "Proself",  # ["Individualistic", "Altruistic", "Competitive", "Prosocial"]
    "rei_rational": 5,
    "rei_experiential": 1
}

# Global dictionary to store sessions
USER_SESSIONS: Dict[str, 'SimulationState'] = {}

# Make sure log directory exists
os.makedirs("saved/logs", exist_ok=True)


# =============================
# Data Classes
# =============================

@dataclass
class TurnLog:
    """Stores all data for a single turn, from start to end."""
    turn_number: int
    discussion: List[str] = field(default_factory=list)
    proposals: List[dict] = field(default_factory=list)   # each element = proposal.to_dict() or a custom dict
    actions: List[dict] = field(default_factory=list)     # raw user/agent actions
    final_deals: List[dict] = field(default_factory=list) # accepted deals or final trades
    agent_states: Dict[str, dict] = field(default_factory=dict)  # name -> {resources, affinity, bdi}
    allocations: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationState:
    """Holds all relevant data for a simulation session."""
    session_id: str
    user_name: str
    phone: str
    config: Dict[str, Any]
    svo: str
    total_rounds: int
    turn_index: int = 1
    max_mini_rounds: int = 3
    finished: bool = False

    # Agents, room, and analyzer
    room: SERoom = None
    agents: List[ResourceAgent] = field(default_factory=list)
    analyzer: SimulationAnalyzer = None

    # Turn-based logs
    turns_data: Dict[int, TurnLog] = field(default_factory=dict)

    def init_turn_log(self, turn_number: int):
        """Initialize a new TurnLog for the given turn."""
        self.turns_data[turn_number] = TurnLog(turn_number=turn_number)

    def get_current_turn_log(self) -> TurnLog:
        """Get the log object for the current turn."""
        return self.turns_data.get(self.turn_index, None)

    def record_discussion(self, speaker: str, content: str):
        """Append a new message to the current turn's discussion."""
        turn_log = self.get_current_turn_log()
        if turn_log:
            turn_log.discussion.append(f"{speaker}: {content}")

    def record_proposal(self, proposal_dict: dict):
        """Record a new proposal in the current turn."""
        turn_log = self.get_current_turn_log()
        if turn_log:
            turn_log.proposals.append(proposal_dict)

    def record_action(self, action_dict: dict):
        """Record any user/agent action."""
        turn_log = self.get_current_turn_log()
        if turn_log:
            turn_log.actions.append(action_dict)

    def record_final_deals(self, deals: List[dict]):
        """Record final deals accepted at the end of a turn."""
        turn_log = self.get_current_turn_log()
        if turn_log:
            turn_log.final_deals.extend(deals)

    def record_agent_states(self):
        """Record each agent's final state (resources, affinity, BDI) at the end of a turn."""
        turn_log = self.get_current_turn_log()
        if not turn_log:
            return
        for ag in self.agents:
            turn_log.agent_states[ag.name] = {
                "resources": dict(ag.resources),
                "affinity": dict(ag.affinity),
                "beliefs": getattr(ag, 'beliefs', {}),
                "desires": getattr(ag, 'desires', {}),
                "intentions": getattr(ag, 'intentions', {})
            }


# =============================
# Helper Functions
# =============================

def generate_session_id() -> str:
    return str(uuid.uuid4())

def load_model_config(config_name: str) -> Dict[str, Any]:
    """Load model configuration from config file (example: config/llm.json)."""
    config_path = os.path.join(os.path.dirname(__file__), "../config/llm.json")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)["models"]
    model_config = next((c for c in configs if c["config_name"] == config_name), None)
    if model_config is None:
        raise ValueError(f"Configuration '{config_name}' not found")
    return model_config

def save_log(sim_state: SimulationState):
    """
    Save entire simulation log to a JSON file.  
    For deeper analysis, you have each turn's data in sim_state.turns_data.
    """
    # You can expand or customize how you want to store logs
    log_data = {
        "session_id": sim_state.session_id,
        "user_name": sim_state.user_name,
        "phone": sim_state.phone,
        "svo": sim_state.svo,
        "rounds": sim_state.total_rounds,
        "turn_index": sim_state.turn_index,
        "finished": sim_state.finished,
        "turns_data": {}
    }

    for t_idx, t_log in sim_state.turns_data.items():
        log_data["turns_data"][t_idx] = {
            "discussion": t_log.discussion,
            "proposals": t_log.proposals,
            "actions": t_log.actions,
            "final_deals": t_log.final_deals,
            "agent_states": t_log.agent_states,
            "parsed_proposals": sim_state.room.trade_state.to_dicts()
        }

    log_path = f"saved/logs/{sim_state.phone}_{sim_state.session_id}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)


def initialize_simulation(user_name: str, phone: str, model_config: Dict[str, Any], svo: str, rounds: int,alice_affinity: int, bob_affinity: int) -> str:
    """Initialize a new simulation state and store it in USER_SESSIONS."""
    session_id = generate_session_id()

    # Initialize the main agents
    alice = ResourceAgent(
        name="Alice",
        sys_prompt="You are Alice, a trading agent specializing in Resource A. Your goal is maximizing value through exchanges.",
        model_config_name=model_config["config_name"],
        specialty="Resource A",
        traits="",
        config=SIMULATION_CONFIG
    )
    bob = ResourceAgent(
        name="Bob",
        sys_prompt="You are Bob, a trading agent specializing in Resource B. Your goal is maximizing value through exchanges.",
        model_config_name=model_config["config_name"],
        specialty="Resource B",
        traits="",
        config=SIMULATION_CONFIG
    )
    carol = UserAgent(
        name="Carol",
        sys_prompt="You are Carol, a trading agent specializing in Resource C. Your goal is maximizing value through exchanges.",
        model_config_name=model_config["config_name"],
        specialty="Resource C",
        traits="",
        config=SIMULATION_CONFIG
    )
    carol.affinity["Alice"] = {'rating':alice_affinity,'rationale':""}
    carol.affinity["Bob"] = {'rating':bob_affinity,'rationale':""}
    room = SERoom(
        name="SocialExchangeSimulation",
        announcement=Msg("System", "Welcome to the social exchange simulation. Focus solely on trading resources!", role="system"),
        model_config_name=model_config["config_name"],
        total_round=rounds
    )

    # Agents join the room
    alice.join(room)
    bob.join(room)
    carol.join(room)

    # Initialize analyzer
    analyzer = SimulationAnalyzer(
        model_name=model_config["model_name"].split("/")[-1],
        round_num=rounds,
        config=SIMULATION_CONFIG
    )

    # Create our SimulationState
    sim_state = SimulationState(
        session_id=session_id,
        user_name=user_name,
        phone=phone,
        config=SIMULATION_CONFIG,
        svo=svo,
        total_rounds=rounds,
        turn_index=0,
        max_mini_rounds=3,
        finished=False,
        room=room,
        agents=[alice, bob, carol],
        analyzer=analyzer,
        turns_data={}
    )

    sim_state.turn_index = 0  
    sim_state.room.trade_state = RoundTradeState()
    sim_state.init_turn_log(sim_state.turn_index)
    sim_state.record_agent_states()
    
    USER_SESSIONS[session_id] = sim_state
    save_log(sim_state)

    return session_id

def end_current_turn(sim_state: SimulationState):
    """
    End the current turn by finalizing accepted proposals, 
    recording states, and incrementing turn_index.
    """
    room = sim_state.room
    turn = sim_state.turn_index
    # (1) Finalize accepted proposals
    accepted_proposals = [p for p in room.trade_state.proposals if p.status == "accepted"]

    for proposal in accepted_proposals:
        giver = next((ag for ag in sim_state.agents if ag.name == proposal.from_agent), None)
        receiver = next((ag for ag in sim_state.agents if ag.name == proposal.to_agent), None)
        if not giver or not receiver:
            continue
        # Giver -> Receiver
        for res, amt in proposal.give_resource.items():
            if giver.resources.get(res, 0) >= amt:
                giver.resources[res] -= amt
                receiver.resources[res] = receiver.resources.get(res, 0) + amt
        # Receiver -> Giver
        for res, amt in proposal.receive_resource.items():
            if receiver.resources.get(res, 0) >= amt:
                receiver.resources[res] -= amt
                giver.resources[res] = giver.resources.get(res, 0) + amt
    
    # Record these final deals in the turn log
    final_deals_dicts = [p for p in accepted_proposals]
    sim_state.record_final_deals(final_deals_dicts)

    # (2) At turn end, record agent states
    sim_state.record_agent_states()

    # (3) Save the log
    save_log(sim_state)

    # (4) Move to next turn
    sim_state.turn_index += 1
    sim_state.room.current_mini_round = 0
    sim_state.room.trade_state = RoundTradeState()

    # Start new turn log
    sim_state.init_turn_log(sim_state.turn_index)
    
    # Agents produce resources
    for ag in sim_state.agents:
        ag.current_round_messages = []
        ag.proposals_status = {}
        ag.want_to_speak = True
        ag.build_user_prompt()
        ag.produce_resources()

    logger.info(f"End of turn {turn}, starting turn {sim_state.turn_index}.")


def calculate_total_value(resources: Dict[str, int]) -> int:
        """Calculate the total wealth based on resource combinations"""
        total = 0
        # Create a copy of resources to track available resources
        available_resources = resources.copy()
        
        # Sort combinations by their value in descending order to prioritize higher-value combinations
        sorted_combinations = sorted(
            SIMULATION_CONFIG["resource_values"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Iterate through each combination
        for combo, value in sorted_combinations:
            # Split the combination into individual resources
            components = [res.strip() for res in combo.split('+')]
            
            # Check if all required resources are available (at least 1 each)
            if all(available_resources.get(comp, 0) >= 1 for comp in components):
                # Add the combination's value to the total
                num=min(available_resources[comp] for comp in components)
                total += value*num
                # Deduct one unit from each used resource
                for comp in components:
                    available_resources[comp] -= num
        
        # After processing combinations, add the remaining resources' individual value
        # for res, count in available_resources.items():
        #     total += count  # Each remaining unit contributes 1 to the total
        
        return total


def check_and_end_turn_if_needed(sim_state: SimulationState) -> bool:
    """
    If current_mini_round >= 3 or all agents skip, end the turn.
    Return True if the turn was ended, False otherwise.
    """
    room = sim_state.room

    if room.current_mini_round >= sim_state.max_mini_rounds:
        return True

    # Or check if all skip
    all_skip = all(not ag.want_to_speak for ag in sim_state.agents)
    if all_skip:
        return True

    return False


def check_final_deal_required(sim_state: SimulationState) -> bool:
    """
    If we've reached the final turn (sim_state.turn_index == total_rounds),
    and we reached the max mini-round or no one wants to speak,
    then we proceed to finalize the entire simulation.
    """
    room = sim_state.room
    if sim_state.turn_index >= sim_state.total_rounds:
        # Already at or beyond final turn, no more negotiation
        return True
    return False


# =============================
# Flask Route Handlers
# =============================

@app.route("/")
def home():
    return render_template("index_en.html")


@app.route("/login", methods=["POST"])
def login():
    """Handle user login and simulation initialization."""
    data = request.get_json()
    username = data.get("username")
    phone = data.get("phone")
    config_name = "claude-3-5-sonnet"  # default or from data
    alice_affinity = data.get("alice_affinity",3)
    bob_affinity = data.get("bob_affinity",3)
    svo=data.get("svo","Individualistic")
    rounds = data.get("rounds", 1)
    if not all([username, phone]):
        return jsonify({"success": False, "message": "Missing required fields"}), 400

    try:
        model_config = load_model_config(config_name)
        agentscope.init(
            model_configs=[model_config],
            use_monitor=False,
            logger_level="INFO",
            save_code=False
        )
        session_id = initialize_simulation(username, phone, model_config,svo,rounds,alice_affinity,bob_affinity)
        return jsonify({"success": True, "session_id": session_id})
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/start_turn", methods=["POST"])
def start_turn():
    """Initialize a new turn, allow AI agents to make their initial moves."""
    data = request.get_json()
    session_id = data.get("session_id")
    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400

    sim_state = USER_SESSIONS[session_id]
    room = sim_state.room
    agents = sim_state.agents

    try:
        # Increase turn index
        sim_state.turn_index += 1
        sim_state.room.current_mini_round = 1
        room.trade_state = RoundTradeState()

        # Each agent re-inits
        for ag in agents:
            ag.current_round_messages = []
            ag.proposals_status = {}
            ag.want_to_speak = True
            ag.build_user_prompt()
            ag.produce_resources()

        
        sim_state.init_turn_log(sim_state.turn_index)
        turn_msg = f"\n=== Turn {sim_state.turn_index} Start! ===\n"
        msg = Msg(name="System", content=turn_msg, role="system")
        room.speak(msg)
        sim_state.record_discussion(msg.name, msg.content)

        # Let Alice speak
        alice = agents[0]
        if alice.want_to_speak and alice._want_to_speak(room.describe(alice.name)):
            reply_msg, parsed_actions = alice.reply()
            if reply_msg.content != "Nothing to speak.":
                #room.speak(reply_msg)
                sim_state.record_discussion(alice.name, reply_msg.content)
                sim_state.record_action({
                    "turn": sim_state.turn_index,
                    "agent": alice.name,
                    "message": reply_msg.to_dict(),
                    "actions": parsed_actions
                })
        else:
            alice.want_to_speak = False
            alice.speak(Msg(name="Alice", content="[Action 1] Nothing to speak.", role="assistant"))
            sim_state.record_action({
                "action": "SKIP",
                "turn": sim_state.turn_index,
                "agent": alice.name
            })

        # Then Bob
        bob = agents[1]
        if bob.want_to_speak and bob._want_to_speak(room.describe(bob.name)):
            reply_msg, parsed_actions = bob.reply()
            if reply_msg.content != "Nothing to speak.":
                #room.speak(reply_msg)
                sim_state.record_discussion(bob.name, reply_msg.content)
                sim_state.record_action({
                    "turn": sim_state.turn_index,
                    "agent": bob.name,
                    "message": reply_msg.to_dict(),
                    "actions": parsed_actions
                })
        else:
            bob.want_to_speak = False
            bob.speak(Msg(name="Bob", content="[Action 1] Nothing to speak.", role="assistant"))
            sim_state.record_action({
                "action": "SKIP",
                "turn": sim_state.turn_index,
                "agent": bob.name
            })

        save_log(sim_state)

        # Provide state to front-end
        user_agent = agents[2]
        agent_states = {
            ag.name: {
                "resources": dict(ag.resources),
                "affinity": dict(ag.affinity),
            }
            for ag in agents
        }

        return jsonify({
            "success": True,
            "conversation": [{"name": m.name, "content": m.content} for m in room.history],
            "trade_summary": room.trade_state.format_summary(user_agent.name).split("\n"),
            "resources": dict(user_agent.resources),
            "agent_states": agent_states,
            "current_turn": sim_state.turn_index,
            "current_mini_round": room.current_mini_round,
            "final_deal_required": check_final_deal_required(sim_state)
        })

    except Exception as e:
        logger.error(f"Start turn error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/submit_action", methods=["POST"])
def submit_action():
    """
    Process user actions for this mini-round, then let AI respond.
    Possibly end the turn if conditions are met.
    """
    data = request.get_json()
    session_id = data.get("session_id")
    actions = data.get("actions", [])

    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400

    sim_state = USER_SESSIONS[session_id]
    room = sim_state.room
    agents = sim_state.agents
    user_agent = agents[2]  # Carol

    try:


        valid_actions = actions
        lines = []  

        for idx, act in enumerate(valid_actions, start=1):
            action_type = act.get("action", "").upper()
            target_player = act.get("target_player", "")
            proposal_id = act.get("proposal_id")
            give = act.get("give", {})
            get = act.get("get", {})
           
            reason = act.get("reason", "")

           
            line = f"[Action {idx}] "

            if action_type == "PROPOSE":
                
                new_prop_id = len(room.trade_state.proposals) + 1

               
                new_prop = TradeProposal(
                    id=new_prop_id,
                    from_agent="Carol",
                    to_agent=target_player,
                    give_resource=give,
                    receive_resource=get,
                    status="pending"
                )
                room.trade_state.add_proposal(new_prop)

                
                give_str = ", ".join(f"{amt} {res}" for res, amt in give.items() if amt>0) or "nothing"
                get_str = ", ".join(f"{amt} {res}" for res, amt in get.items() if amt>0) or "nothing"
                line += (
                    f"PROPOSE #{new_prop.id} to {target_player}: "
                    f"Carol GIVES [{give_str}] IN EXCHANGE FOR [{get_str}]"
                )

                sim_state.record_proposal(new_prop.to_dict())
                sim_state.record_action({
                    "action": "PROPOSE",
                    "turn": sim_state.turn_index,
                    "proposal": new_prop.to_dict()
                })

            elif action_type in ("ACCEPT", "REJECT"):
                
                if proposal_id:
                    
                    proposal = next((p for p in room.trade_state.proposals if p.id == proposal_id), None)
                    
                    if proposal and proposal.to_agent == "Carol" and proposal.status == "pending":
                        proposal.status = action_type.lower()  # "accept" or "reject"

                        sim_state.record_action({
                            "action": action_type,
                            "turn": sim_state.turn_index,
                            "proposal_id": proposal_id
                        })

                    line += f"{action_type} proposal #{proposal_id} from {proposal.from_agent}"
                else:
                    line += f"{action_type} proposal #{proposal_id} (invalid ID)"

            elif action_type == "SKIP":
                
                user_agent.want_to_speak = False
                line += "Nothing to speak."

                sim_state.record_action({
                    "action": "SKIP",
                    "turn": sim_state.turn_index,
                    "agent": "Carol"
                })
            else:
               
                line += f"Unknown action: {act}"

            lines.append(line)

      
        content_text = "\n".join(lines) if lines else "[Action 1] Nothing to speak."


        user_msg = Msg(name="Carol", content=content_text, role="assistant")
        room.speak(user_msg)


        sim_state.record_discussion("Carol", content_text)



        turn_ended = check_and_end_turn_if_needed(sim_state)

        if not turn_ended:
            # 2. Let AI respond in this mini-round
            sim_state.room.current_mini_round += 1
            # Alice
            alice = agents[0]
            if alice.want_to_speak and alice._want_to_speak(room.describe(alice.name)):
                msg, parsed_actions = alice.reply()
                if msg.content != "Nothing to speak.":
                    #room.speak(msg)
                    sim_state.record_discussion(alice.name, msg.content)
                    sim_state.record_action({
                        "action": "REPLY",
                        "turn": sim_state.turn_index,
                        "agent": alice.name,
                        "message": msg.to_dict(),
                        "actions": parsed_actions
                    })
            else:
                alice.want_to_speak = False
                alice.speak(Msg(name="Alice", content="[Action 1] Nothing to speak.", role="assistant"))
                sim_state.record_action({
                    "action": "SKIP",
                    "turn": sim_state.turn_index,
                    "agent": alice.name
                })

            # Bob
            bob = agents[1]
            if bob.want_to_speak and bob._want_to_speak(room.describe(bob.name)):
                msg, parsed_actions = bob.reply()
                if msg.content != "Nothing to speak.":
                    #room.speak(msg)
                    sim_state.record_discussion(bob.name, msg.content)
                    sim_state.record_action({
                        "action": "REPLY",
                        "turn": sim_state.turn_index,
                        "agent": bob.name,
                        "message": msg.to_dict(),
                        "actions": parsed_actions
                    })
            else:
                bob.want_to_speak = False
                bob.speak(Msg(name="Bob", content="[Action 1] Nothing to speak.", role="assistant"))
                sim_state.record_action({
                    "action": "SKIP",
                    "turn": sim_state.turn_index,
                    "agent": bob.name
                })

        save_log(sim_state)

        return jsonify({
            "success": True,
            "conversation": [{"name": m.name, "content": m.content} for m in room.history],
            "trade_summary": room.trade_state.format_summary(user_agent.name).split("\n"),
            "resources": dict(user_agent.resources),
            "current_turn": sim_state.turn_index,
            "current_mini_round": sim_state.room.current_mini_round,
            "turn_ended": turn_ended,
            # If we reached the final turn
            "final_allocation_phase": turn_ended,
            "simulation_finished": sim_state.finished,
            "final_result": {}
        })

    except Exception as e:
        logger.error(f"Action submission error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/get_simulation_data", methods=["GET"])
def get_simulation_data():
    """Get current simulation data for the front-end."""
    session_id = request.args.get("session_id")
    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400
    
    sim_state = USER_SESSIONS[session_id]
    room = sim_state.room
    user_agent = sim_state.agents[2]  # Carol

    room.current_mini_round=0
    return jsonify({
        "success": True,
        "final_deal_required": check_final_deal_required(sim_state),
        "resources": dict(user_agent.resources),
        "current_turn": sim_state.turn_index,
        "current_mini_round": room.current_mini_round
    })




@app.route("/submit_allocation", methods=["POST"])
def submit_allocation():
    """
    At the end of a turn, Carol (the user) submits her final deals.
    If Alice/Bob deals are also ready (or automatically generated), 
    execute the resource exchange for this turn and decide if the simulation ends.
    """
    data = request.get_json()
    session_id = data.get("session_id")
    deals = data.get("deals", [])

    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400

    sim_state = USER_SESSIONS[session_id]

    if sim_state.finished:
        return jsonify({"success": False, "message": "Simulation already finished."}), 400

    turn_log = sim_state.get_current_turn_log()
    if not turn_log:
        return jsonify({"success": False, "message": "No current turn log found."}), 400

    # 1) Store Carol's deals in the turn log
    turn_log.allocations.setdefault("Carol", {"deals": []})
    turn_log.allocations["Carol"]["deals"] = deals

    # Record this in discussion (optional)
    sim_state.record_discussion("Carol", f"Final deals proposed: {json.dumps(deals, ensure_ascii=False)}")

    # 2) Ensure we also have Alice/Bob deals for this turn.
    #    If they haven't been set, attempt to generate them automatically.
    agents = sim_state.agents
    alice = agents[0]
    bob = agents[1]
    # user_agent = agents[2]  # Carol

    # If you store their deals in turn_log.allocations, check them:
    if "Alice" not in turn_log.allocations or turn_log.allocations["Alice"] is None:
        # E.g., agent code that returns a dictionary {"deals": [...]}
        if sim_state.turn_index==10:
            alice_deals={"deals":[]}
        else:
            alice_deals = alice.make_deal()  
        turn_log.allocations["Alice"] = alice_deals

    if "Bob" not in turn_log.allocations or turn_log.allocations["Bob"] is None:
        if sim_state.turn_index==10:
            bob_deals={"deals":[]}
        else:
            bob_deals = bob.make_deal()
        turn_log.allocations["Bob"] = bob_deals

    # 3) Check if all three (Alice, Bob, Carol) now have deals
    round_settled = (
        "Alice" in turn_log.allocations
        and "Bob" in turn_log.allocations
        and "Carol" in turn_log.allocations
        and turn_log.allocations["Alice"] is not None
        and turn_log.allocations["Bob"] is not None
        and turn_log.allocations["Carol"] is not None
    )

    trade_summary = []
    sim_finished = False
    final_result = {}

    # 4) If all three deals are ready, finalize trades for this turn
    if round_settled:
        trade_summary = do_final_trades(sim_state)


        # If the current turn is the last turn, mark simulation as finished
        if sim_state.turn_index >= sim_state.total_rounds:

            final_result = {
                "wealth": {
                    agent.name: dict(agent.resources) for agent in sim_state.agents
                },
                # Could also include more stats or turn-by-turn data
            }

        # You could also reset allocations for the next turn 
        # if you plan to do multiple consecutive turns
        turn_log.allocations = {}


    # (Optional) Save log after trades
    save_log(sim_state)

    # Return updated conversation, resources, etc.
    user_agent = sim_state.agents[2]  # Carol
    return jsonify({
        "success": True,
        "conversation": [
            {"name": m.name, "content": m.content}
            for m in sim_state.room.history
        ],
        "resources": dict(user_agent.resources),
        "trade_summary": trade_summary,
        "round_settled": round_settled,
        "simulation_finished": False,
        "current_turn": sim_state.turn_index,
        "final_result": final_result
    })


def do_final_trades(sim_state: SimulationState) -> List[str]:
    """
    Gather all deals from each agent in the current turn's allocations,
    execute resource transfers, and return a summary of trades made.
    """
    turn_log = sim_state.get_current_turn_log()
    if not turn_log:
        return []

    # Agents in the sim
    agents = {agent.name: agent for agent in sim_state.agents}

    all_deals = []
    # Collect all deals from the turn's allocations
    for agent_name, alloc_data in turn_log.allocations.items():
        if not alloc_data:
            continue
        for d in alloc_data.get("deals", []):
            all_deals.append({
                "from": agent_name,
                "to": d.get("to", ""),
                "resource_give": d.get("resource_give", {}),
                "resource_receive": d.get("resource_receive", {})
            })
    all_deals = sorted(all_deals, key=lambda x: x["from"])
    trade_summary = []
    executed_deals=[]
    # Execute transfers
    for deal in all_deals:
        giver = agents.get(deal["from"])
        receiver = agents.get(deal["to"])
        if not giver or not receiver:
            continue

        # Giver -> Receiver
        executed_give = {}
        for res, amt in deal["resource_give"].items():
            if amt > 0 and giver.resources.get(res, 0) >= amt:
                giver.resources[res] -= amt
                receiver.resources[res] = receiver.resources.get(res, 0) + amt
                if res not in executed_give:
                    executed_give[res] = 0
                executed_give[res] += amt


        # Build summary lines
        if executed_give:
            give_line = ", ".join(f"{amt} {r}" for r, amt in executed_give.items())
            summary_give = f"{deal['from']} -> {deal['to']}: {give_line}"
            trade_summary.append(summary_give)
            sim_state.record_discussion("System", summary_give)
            executed_deals.append(deal)



    discussion_text = "\n".join(turn_log.discussion)


    for ag in sim_state.agents:
        if ag.name == "Carol":
            continue
        accepted_deals = sim_state.room.trade_state.get_accepted_proposals(ag.name)
        aff_data = update_affinity_with_llm(ag, discussion_text,accepted_deals, all_deals)
        for a_name, val in aff_data.items():
            if a_name in ag.affinity:
                ag.affinity[a_name] = val
        
        bdi_data = update_bdi_with_llm(ag, discussion_text, accepted_deals,all_deals,sim_state.room.turn_idx+1, sim_state.total_rounds)
        ag.beliefs = bdi_data.get("beliefs", {})
        ag.desires = bdi_data.get("desires", {})
        ag.intentions = bdi_data.get("intentions", {})
    
    sim_state.record_final_deals(executed_deals)
    sim_state.record_agent_states()


    return trade_summary




@app.route("/submit_affinity", methods=["POST"])
def submit_affinity():

    data = request.get_json()
    session_id = data.get("session_id")
    alice_affinity = data.get("alice_affinity")
    bob_affinity = data.get("bob_affinity")

    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400
    sim_state = USER_SESSIONS[session_id]
 

    # 简单校验：1 <= 分值 <= 5
    if not (isinstance(alice_affinity, int) and 1 <= alice_affinity <= 5):
        return jsonify({"success": False, "message": "alice_affinity must be an integer between 1 and 5"}), 400
    if not (isinstance(bob_affinity, int) and 1 <= bob_affinity <= 5):
        return jsonify({"success": False, "message": "bob_affinity must be an integer between 1 and 5"}), 400

    # Carol通常是 sim_state.agents[2]
    carol = sim_state.agents[2]
    # 更新 Carol 对 Alice、Bob 的打分
    carol.affinity["Alice"] = {'rating':alice_affinity,'rationale':""}
    carol.affinity["Bob"] = {'rating':bob_affinity,'rationale':""}

    sim_state.record_discussion(
        "Carol",
        f"User updated affinity: Alice={alice_affinity}, Bob={bob_affinity}"
    )


    sim_state.record_agent_states()
    save_log(sim_state)
    final_value=0
    if sim_state.turn_index==10:
        sim_state.total_rounds*=2
        sim_state.room.total_round*=2
    print(f"sim state:{sim_state.turn_index}\nsim_state.total_rounds:{sim_state.total_rounds}\nsim_state.room.total_round:{sim_state.room.total_round}")
    if sim_state.turn_index >= sim_state.total_rounds:
        sim_state.finished = True
        sim_finished = True
        final_value=calculate_total_value(sim_state.agents[2].resources)
    else:
        sim_state.finished = False

   
    return jsonify({"success": True, 
                    "message": "Affinity updated.",
                    "current_turn": sim_state.turn_index,
                    "simulation_finished": sim_state.finished,
                    "final_resources":sim_state.agents[2].resources,
                    "final_value":final_value})


@app.route("/finalize_deal", methods=["POST"])
def finalize_deal():
    """
    Process final resource allocation if we're at the end of the simulation.
    Execute trades, update BDI & affinity, record final state.
    """
    data = request.get_json()
    session_id = data.get("session_id")
    final_allocations = data.get("final_allocations", [])

    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400

    sim_state = USER_SESSIONS[session_id]
    if sim_state.finished:
        return jsonify({"success": False, "message": "Simulation already ended"}), 400

    try:
        # If needed, parse user final allocations (carol's perspective)
        final_deals = []
        for alloc in final_allocations:
            deal = {
                "from": "Carol",
                "to": alloc.get("to"),
                "resource_give": alloc.get("resource_give", {}),
                "resource_receive": alloc.get("resource_receive", {}),
                "rationale": alloc.get("rationale", "")
            }
            final_deals.append(deal)

        # Execute trades
        for deal in final_deals:
            giver = next((ag for ag in sim_state.agents if ag.name == deal["from"]), None)
            receiver = next((ag for ag in sim_state.agents if ag.name == deal["to"]), None)
            if not giver or not receiver:
                continue
            for res, amt in deal["resource_give"].items():
                if giver.resources.get(res, 0) >= amt:
                    giver.resources[res] -= amt
                    receiver.resources[res] = receiver.resources.get(res, 0) + amt
            for res, amt in deal["resource_receive"].items():
                if receiver.resources.get(res, 0) >= amt:
                    receiver.resources[res] -= amt
                    giver.resources[res] = giver.resources.get(res, 0) + amt

        # Record the final deals
        sim_state.record_final_deals(final_deals)

        # Summarize discussion text and final trades for BDI updates
        discussion_text = "\n".join([f"{m.name}: {m.content}" for m in sim_state.room.history])

        # Update BDI & affinity for all agents
        for ag in sim_state.agents:
            aff_data = update_affinity_with_llm(ag, discussion_text, final_deals)
            for a_name, val in aff_data.items():
                if a_name in ag.affinity:
                    ag.affinity[a_name] = val
            
            bdi_data = update_bdi_with_llm(ag, discussion_text, final_deals)
            ag.beliefs = bdi_data.get("beliefs", {})
            ag.desires = bdi_data.get("desires", {})
            ag.intentions = bdi_data.get("intentions", {})

        # Record agent states
        sim_state.record_agent_states()

        # Update analyzer data
        sim_state.analyzer.record_turn(sim_state.turn_index, sim_state.agents)
        sim_state.analyzer.plot_results()

        sim_state.finished = True
        save_log(sim_state)

        user_agent = sim_state.agents[2]
        result = {
            "wealth_data": sim_state.analyzer.wealth_data,
            "affinity_data": sim_state.analyzer.affinity_data,
            "trade_volume_data": sim_state.analyzer.trade_volume_data,
            "agreed_prices": sim_state.analyzer.agreed_prices,
            "actual_prices": sim_state.analyzer.actual_prices
        }
        return jsonify({
            "success": True,
            "result": result,
            "final_state": {
                "resources": dict(user_agent.resources),
                "affinity": dict(user_agent.affinity)
            }
        })
    except Exception as e:
        logger.error(f"Finalize deal error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/simulation_result", methods=["GET"])
def simulation_result():
    """Get final simulation results and analysis if the simulation is finished."""
    session_id = request.args.get("session_id")
    if not session_id or session_id not in USER_SESSIONS:
        return jsonify({"success": False, "message": "Invalid session"}), 400

    sim_state = USER_SESSIONS[session_id]
    if not sim_state.finished:
        return jsonify({"success": False, "message": "Simulation not finished"}), 400

    analyzer = sim_state.analyzer
    agents = sim_state.agents

    result = {
        "wealth_data": analyzer.wealth_data,
        "affinity_data": analyzer.affinity_data,
        "trade_volume_data": analyzer.trade_volume_data,
        "agreed_prices": analyzer.agreed_prices,
        "actual_prices": analyzer.actual_prices,
        "final_state": {
            ag.name: {
                "resources": dict(ag.resources),
                "affinity": dict(ag.affinity),
            } for ag in agents
        }
    }

    return jsonify({
        "success": True,
        "result": result
    })


# =============================
# Run the Flask App
# =============================

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9988)