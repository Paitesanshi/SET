import os
import json
import argparse
from typing import List, Dict, Any,Tuple
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime

import agentscope
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonObjectParser
from envs.seroom import SERoom, ChatRoomAgent
from dataclasses import dataclass, asdict
from collections import defaultdict
# =========================
# Global Configuration
# =========================


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
        "Resource A":1,
        "Resource B":1,
        "Resource C":1,
        "Resource A + Resource B": 4,
        "Resource A + Resource C": 4,
        "Resource B + Resource C": 4,
        "Resource A + Resource B + Resource C": 9
    },
    "svo":"Proself",#["Proself", "Prosocial"]
    "rei_rational":5,
    "rei_experiential":1
}

ALL_AGENTS = ["Alice", "Bob", "Carol"]

# Used to store the entire simulation log
GLOBAL_LOG = {
    "turns": []
}

TIMESTAMP=datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class TradeProposal:
    """Single trade proposal data"""
    id: int  # Unique identifier for the proposal
    from_agent: str
    to_agent: str
    give_resource: Dict[str, int]       # 改为 dict
    receive_resource: Dict[str, int]    # 改为 dict
    status: str = "pending"            # "pending", "accepted", "rejected"

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeProposal":
        """Create proposal from dictionary"""
        return cls(**data)
        
    def to_json(self) -> str:
        """Convert proposal to JSON string"""
        return json.dumps(self.to_dict())
        
    @classmethod
    def from_json(cls, json_str: str) -> "TradeProposal":
        """Create proposal from JSON string"""
        return cls.from_dict(json.loads(json_str))


class RoundTradeState:
    """Tracks all trades for a negotiation round"""
    def __init__(self):
        self.proposals: List[TradeProposal] = []
        self.id_counter: int = 1  # Initialize proposal ID counter
    
    def add_proposal(self, proposal: TradeProposal):
        proposal.id = self.id_counter  # Assign unique ID
        self.id_counter += 1
        self.proposals.append(proposal)
    
    def clear(self):
        self.proposals = []

    
    def to_dicts(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.proposals]
    
    def get_agent_proposals(self, agent_name: str) -> List[TradeProposal]:
        """Get all proposals involving this agent"""
        return [
            p for p in self.proposals 
            if p.from_agent == agent_name or p.to_agent == agent_name
        ]
    
    def get_pending_proposals(self, agent_name: str) -> List[TradeProposal]:
        """Get pending proposals that this agent needs to respond to"""
        return [
            p for p in self.proposals 
            if p.to_agent == agent_name and p.status == "pending"
        ]
    

    def format_summary(self, agent_name: str) -> str:
        """Format trade status for an agent with clear trade descriptions"""
        summary = []
        
        # Proposals made by this agent
        my_proposals = [p for p in self.proposals if p.from_agent == agent_name]
        if my_proposals:
            summary.append(f"{agent_name}'s proposals:")
            for p in my_proposals:
                give_str = ", ".join(f"{amt} {res}" for res, amt in p.give_resource.items())
                recv_str = ", ".join(f"{amt} {res}" for res, amt in p.receive_resource.items())
                summary.append(
                f"- ID {p.id}: {p.from_agent}'s [{give_str}] "
                f"↔ {p.to_agent}'s [{recv_str}] "
                f"| Status: {p.status}"
                )
        
        # Proposals to this agent  
        received = [p for p in self.proposals if p.to_agent == agent_name]
        if received:
            summary.append(f"\nProposals to {agent_name}:")
            for p in received:
                give_str = ", ".join(f"{amt} {res}" for res, amt in p.give_resource.items())
                recv_str = ", ".join(f"{amt} {res}" for res, amt in p.receive_resource.items())
                summary.append(
                f"- ID {p.id}: {p.from_agent}'s [{give_str}] "
                f"↔ {p.to_agent}'s [{recv_str}] "
                f"| Status: {p.status}"
                )
        
        return "\n".join(summary)
    
    def get_accepted_proposals(self, agent_name: str) -> List[dict]:
        """
        Get accepted proposals for an agent, merging bidirectional proposals between the same agents.
        Combines give_resource and receive_resource when merging proposals.
        """
        def merge_resource_dicts(d1: Dict[str, int], d2: Dict[str, int]) -> Dict[str, int]:
            result = d1.copy()
            for resource, amount in d2.items():
                result[resource] = result.get(resource, 0) + amount
            return result

        accepted = [
            p for p in self.proposals 
            if (p.to_agent == agent_name or p.from_agent == agent_name)
            and p.status == "accepted"
        ]
        
        merged = {}
        
        for proposal in accepted:
            agents = tuple(sorted([proposal.from_agent, proposal.to_agent]))
            
            if agents not in merged:
                merged[agents] = {
                    'id': proposal.id,
                    'from_agent': agents[0],
                    'to_agent': agents[1],
                    'give_resource': proposal.give_resource.copy(),
                    'receive_resource': proposal.receive_resource.copy(),
                    'status': proposal.status
                }
            else:
                # If direction is reversed, swap give/receive resources before merging
                if proposal.from_agent != merged[agents]['from_agent']:
                    merged[agents]['give_resource'] = merge_resource_dicts(
                        merged[agents]['give_resource'], 
                        proposal.receive_resource
                    )
                    merged[agents]['receive_resource'] = merge_resource_dicts(
                        merged[agents]['receive_resource'], 
                        proposal.give_resource
                    )
                else:
                    merged[agents]['give_resource'] = merge_resource_dicts(
                        merged[agents]['give_resource'], 
                        proposal.give_resource
                    )
                    merged[agents]['receive_resource'] = merge_resource_dicts(
                        merged[agents]['receive_resource'], 
                        proposal.receive_resource
                    )
                        
        return list(merged.values())


def update_bdi_with_llm(agent, history: str, accepted_proposals: List[Dict], trades: List[Dict],current_round:int,total_rounds:int) -> Dict:
    """
    Use LLM to update agent's BDI framework by analyzing current state, trades and relationships.
    """
    parser = MarkdownJsonObjectParser()
            
    promised_trades = "=== PROMISED EXCHANGES ===\n"
    if len(accepted_proposals) > 0:
        
        trade_pairs = {}
        for p in accepted_proposals:
            if p["from_agent"] == agent.name or p["to_agent"] == agent.name:
                pair_key = tuple(sorted([p["from_agent"], p["to_agent"]]))
                if pair_key not in trade_pairs:
                    trade_pairs[pair_key] = {
                        "agent1": pair_key[0],
                        "agent2": pair_key[1],
                        "agent1_promised": defaultdict(int), 
                        "agent2_promised": defaultdict(int)
                    }
                if p["from_agent"] == pair_key[0]:
                    for resource, amount in p["give_resource"].items():
                        trade_pairs[pair_key]["agent1_promised"][resource] += amount
                    for resource, amount in p["receive_resource"].items():
                        trade_pairs[pair_key]["agent2_promised"][resource] += amount
                else:
                    for resource, amount in p["give_resource"].items():
                        trade_pairs[pair_key]["agent2_promised"][resource] += amount
                    for resource, amount in p["receive_resource"].items():
                        trade_pairs[pair_key]["agent1_promised"][resource] += amount
        
        for pair_info in trade_pairs.values():
            promised_trades += (
                f"Exchange Between: {pair_info['agent1']} and {pair_info['agent2']}\n"
                f"{pair_info['agent1']} Total Promised to Give: {dict(pair_info['agent1_promised'])}\n"
                f"{pair_info['agent2']} Total Promised to Give: {dict(pair_info['agent2_promised'])}\n"
                "------------------------\n"
            )
    else:
        promised_trades += "No exchanges have been promised.\n"
    actual_trades = "=== ACTUAL EXCHANGES ===\n"

    executed_pairs = {}
    for t in trades:
        if t["from"] == agent.name or t["to"] == agent.name:
            pair_key = tuple(sorted([t["from"], t["to"]]))
            if pair_key not in executed_pairs:
                executed_pairs[pair_key] = {
                    "agent1": pair_key[0],
                    "agent2": pair_key[1],
                    "agent1_gave": defaultdict(int),
                    "agent2_gave": defaultdict(int)
                }
            if t["from"] == pair_key[0]:
                for resource, amount in t["resource_give"].items():
                    executed_pairs[pair_key]["agent1_gave"][resource] += amount
            else:
                for resource, amount in t["resource_give"].items():
                    executed_pairs[pair_key]["agent2_gave"][resource] += amount

    for pair_info in executed_pairs.values():
        actual_trades += (
            f"Exchange Between: {pair_info['agent1']} and {pair_info['agent2']}\n"
            f"{pair_info['agent1']} Total Actually Gave: {dict(pair_info['agent1_gave'])}\n"
            f"{pair_info['agent2']} Total Actually Gave: {dict(pair_info['agent2_gave'])}\n"
            "------------------------\n"
        )
    user_prompt = f"""
Please analyze the current state and update your BDI framework based on:
1. Conversation history
2. Promised trades
3. Actual executed trades
4. Current resource holdings: {json.dumps(agent.resources)}
5. Current round: {current_round}/{total_rounds}

**Core Strategic Anchors**  
① ABC Balance Priority: Maintain progression toward A+B+C=9 combination  
② Trust Gradient: Partners showing consistent promise-keeping get priority  
③ Phase Awareness:  
   Early Phase → Relationship probing with small trades  
   Mid Phase → Optimizing complementary resource exchanges  
   Late Phase → Securing final combination requirements  

**Analysis Framework**  
[Beliefs] (Observed Patterns)  
- Resource status indicating: [Your inference about resource gaps]  
- Behavioral patterns showing: [Trustworthiness assessment]  

[Desires] (Strategic Goals)  
- Primary objective: [Phase-specific main focus]  
- Secondary objective: [Backup/supporting goal]  

[Intentions] (Action Plan)  
- Next-step trades: [Specific resource exchange proposal]  
- Risk buffer: [Natural consequence of observed patterns]  

Return analysis in this format:  
{{
    "beliefs": "[Concise pattern recognition]",  
    "desires": "[Hierarchical objectives]",  
    "intentions": "[Action steps with inherent safeguards]"  
}}
"""
    user_prompt=f"{agent.build_user_prompt()}\nConversation:\n{history}\n{promised_trades}\n{actual_trades}\n{user_prompt}"
    prompt = agent.model.format(
        Msg("system", agent.sys_prompt, role="system"),
        # Msg("assistant", f"Conversation:\n{history}", role="assistant"), 
        # Msg("assistant", f"{promised_trades}\n{actual_trades}\n", role="assistant"),
        Msg("user", user_prompt + parser.format_instruction, role="user")
    )

    try:
        resp = agent.model(prompt)
        data = parser.parse(resp).parsed
        return data
    except Exception as e:
        logger.warning(f"{agent.name} update_bdi_with_llm failed: {e}")
        return {"beliefs": {}, "desires": {}, "intentions": {}}

# =========================
# LLM Helper Function: Update Affinity
# =========================


def update_affinity_with_llm(agent, history: str, accepted_proposals: List[Dict], trades: List[Dict]) -> Dict:
    """
    Use LLM to update affinity scores by analyzing promised vs actual trades.
    """
    parser = MarkdownJsonObjectParser()
    

    promised_trades = "=== PROMISED EXCHANGES ===\n"
    if len(accepted_proposals) > 0:
        trade_pairs = {}
        for p in accepted_proposals:
            if p["from_agent"] == agent.name or p["to_agent"] == agent.name:
                pair_key = tuple(sorted([p["from_agent"], p["to_agent"]]))
                if pair_key not in trade_pairs:
                    trade_pairs[pair_key] = {
                        "agent1": pair_key[0],
                        "agent2": pair_key[1],
                        "agent1_promised": defaultdict(int),  
                        "agent2_promised": defaultdict(int)
                    }
                if p["from_agent"] == pair_key[0]:
                    for resource, amount in p["give_resource"].items():
                        trade_pairs[pair_key]["agent1_promised"][resource] += amount
                    for resource, amount in p["receive_resource"].items():
                        trade_pairs[pair_key]["agent2_promised"][resource] += amount
                else:
                    for resource, amount in p["give_resource"].items():
                        trade_pairs[pair_key]["agent2_promised"][resource] += amount
                    for resource, amount in p["receive_resource"].items():
                        trade_pairs[pair_key]["agent1_promised"][resource] += amount
        
        for pair_info in trade_pairs.values():
            promised_trades += (
                f"Exchange Between: {pair_info['agent1']} and {pair_info['agent2']}\n"
                f"{pair_info['agent1']} Total Promised to Give: {dict(pair_info['agent1_promised'])}\n"
                f"{pair_info['agent2']} Total Promised to Give: {dict(pair_info['agent2_promised'])}\n"
                "------------------------\n"
            )
    else:
        promised_trades += "No exchanges have been promised.\n"

 
    actual_trades = "=== ACTUAL EXCHANGES ===\n"

    executed_pairs = {}
    for t in trades:
        if t["from"] == agent.name or t["to"] == agent.name:
            pair_key = tuple(sorted([t["from"], t["to"]]))
            if pair_key not in executed_pairs:
                executed_pairs[pair_key] = {
                    "agent1": pair_key[0],
                    "agent2": pair_key[1],
                    "agent1_gave": defaultdict(int),
                    "agent2_gave": defaultdict(int)
                }
            if t["from"] == pair_key[0]:
                for resource, amount in t["resource_give"].items():
                    executed_pairs[pair_key]["agent1_gave"][resource] += amount
            else:
                for resource, amount in t["resource_give"].items():
                    executed_pairs[pair_key]["agent2_gave"][resource] += amount

    for pair_info in executed_pairs.values():
        actual_trades += (
            f"Exchange Between: {pair_info['agent1']} and {pair_info['agent2']}\n"
            f"{pair_info['agent1']} Total Actually Gave: {dict(pair_info['agent1_gave'])}\n"
            f"{pair_info['agent2']} Total Actually Gave: {dict(pair_info['agent2_gave'])}\n"
            "------------------------\n"
        )
    


    user_prompt = """
Update affinity ratings (1-5) by evaluating both trust patterns and tangible benefits:

# Core Evaluation Dimensions
**Trust Dynamics** (Relationship Foundation):
- Major Betrayal: Significant under-delivery without justification
- Repeated Under-performance: Pattern of unmet commitments
- Recovery Attempts: Proactive compensation for past failures
- Consistent Reliability: Sustained promise fulfillment

**Benefit Sensitivity** (Self-Interest Focus):
- Value Surplus: Over-delivery beyond commitments
- Strategic Concessions: Unprompted favorable terms
- Hidden Generosity: Non-transactional resource sharing
- Opportunity Cost: Alternatives sacrificed for your benefit

# Behavioral Thresholds
▲ Upgrade Triggers:
- Spontaneous high-value gift (unrequested)
- Critical support during resource shortage
- Consistently exceeding promises (3+ rounds)

▼ Downgrade Triggers:
- Opportunistic exploitation during crisis
- Pattern of ambiguous commitments
- Repeated last-minute term changes

# Adaptive Rating Guide
| Level | Interaction Style           | Key Characteristics                  |
|-------|-----------------------------|---------------------------------------|
| 1     | Transactional Enforcement   | Demands collateral, verifies all terms|
| 2     | Cautious Reciprocity        | Limited credit, phased exchanges     |
| 3     | Balanced Partnership        | Market-standard terms with flexibility|
| 4     | Value-Added Collaboration   | Allows payment cycles, shares insights|
| 5     | Synergistic Alliance         | Joint optimization, resource pooling |

# Response Requirements
Return JSON with:
- `rating` (1-5)
- `rationale` (60-90 words) covering:
  1. Primary behavioral pattern observed
  2. Benefit/risk balance assessment
  3. Recommended engagement strategy

Example:
{
  "TradeMaster": {
    "rating": 4,
    "rationale": "Demonstrated strategic generosity by sharing rare Resource X unprompted during shortage (Benefit+). Maintained 93% commitment accuracy over 5 rounds (Trust+). Recommended strategy: Prioritize collaborative projects while maintaining standard verification protocols."
  }
}
"""

    #history=agent.room.get_history()
    user_prompt=f"{agent.build_user_prompt()}\nConversation History:\n{history}\nCurrent Round:\n{promised_trades}\n{actual_trades}\n{user_prompt}"
    prompt = agent.model.format(
        Msg("system", agent.sys_prompt, role="system"),
        Msg("user", user_prompt + parser.format_instruction, role="user")
    )
    
    try:
        resp = agent.model(prompt)
        data = parser.parse(resp).parsed
        return data
    except Exception as e:
        logger.warning(f"{agent.name} update_affinity_with_llm failed: {e}")
        return {"affinity": {}}

# =========================
# ResourceAgent Class
# =========================


class ResourceAgent(ChatRoomAgent):
    """
    1) LLM generates dialogue
    2) LLM parses proposals
    3) LLM strategizes based on accepted proposals during make_deal phase
    4) LLM updates BDI & affinity at the end of each turn
    """
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        specialty: str,
        traits: str,
        config: Dict = SIMULATION_CONFIG,
        **kwargs
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            **kwargs
        )
        self.specialty = specialty
        self.traits = traits
        self.config = config

        self.resources = dict(self.config["initial_resources"])
        self.affinity = {a:{ 'rating':3, 'rationale':'' }for a in ALL_AGENTS if a != self.name}
        self.svo = self.config.get("svo", "Proself") #["Proself", "Prosocial"]
        self.rei_rational = self.config.get("rei_rational", 3)
        self.rei_experiential = self.config.get("rei_experiential", 3)
        # BDI components
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}

        # Store proposals (offer/accepted/rejected) => "accepted"/"rejected"/"pending"
        self.proposals_status = {}
        self.current_round_messages = []
        self.want_to_speak = True

    def produce_resources(self):
        """Produce resources based on specialization"""
        if self.specialty in self.resources:
            self.resources[self.specialty] += self.config["resources"][self.specialty]["production"]
        # No resource consumption, so no deduction

    def build_user_prompt(self):
       
        prompt = [
            f"Resources: {self._format_resources()}",
            
            
            self._build_decision_profile(),
            
            
            self._build_strategic_framework(),
            
            
            self._build_value_system()
        ]
        return "\n\n".join(prompt)

    def _format_resources(self):
        
        return ", ".join([f"{k}:{v}" for k,v in self.resources.items()])

    def _build_decision_profile(self):
        
        svo_profile = {
            "Prosocial": {
                "traits": [
                    "Mutual benefit focus",
                    "Strategic cooperation",
                    "Long-term relationship building",
                ],
                "trade_style": "Propose reciprocal exchanges with clear mutual gains. Always keep commitment.",
            },
            "Proself": {
                "traits": [
                    "Outcome maximization",
                    "Strategic flexibility",
                    "Self-interest priority",
                ],
                "trade_style": "Negotiate advantageous terms with contingency planning",
            },
        }[self.svo]

        rei_rational = [
            "(Heuristic decisions)", 
            "(Balanced analysis)", 
            "(Strategic evaluation)"
        ][min(2, self.rei_rational//2)]

        rei_experiential = [
            "(Analytical focus)", 
            "(Intuition-guided)", 
            "(Expert intuition)"
        ][min(2, self.rei_experiential//2)]

        return (
            "=== Decision Profile ===\n"
            f"SVO: {self.svo}\n"
            f"- Behavioral traits: {', '.join(svo_profile['traits'])}\n"
            f"- Trade approach: {svo_profile['trade_style']}\n\n"
            "REI Balance:\n"
            f"Rational {self.rei_rational}/5 {rei_rational}\n"
            f"Experiential {self.rei_experiential}/5 {rei_experiential}"
        )

    def _build_strategic_framework(self):
        
        if self.room.turn_idx/self.room.total_round<0.3:
            phase=0
        elif self.room.turn_idx/self.room.total_round<0.6:
            phase=1
        else:
            phase=2
        phase_desc = [
            "Early: Relationship building with test trades",
            "Mid: Optimizing resource combinations", 
            "Late: Securing final requirements"
        ][phase]

        return (
            "=== Strategic Framework ===\n"
            f"Phase: {phase_desc}\n"
            "BDI Analysis Structure:\n"
            "Beliefs (Current Observations):\n"
            f"- {self.beliefs}\n\n"
            "Desires (Strategic Goals):\n"
            f"- {self.desires}\n\n"
            "Intentions (Action Plan):\n"
            f"- {self.intentions}"
        )

    def _build_value_system(self):
        
        base_values = [f"{k}={v}" for k,v in self.config['resource_values'].items() if '+' not in k]
        combos = [f"{k}={v}" for k,v in self.config['resource_values'].items() if '+' in k]

        return (
            "=== Value System ===\n"
            "Base Values:\n- " + "\n- ".join(base_values) + "\n\n"
            "Combinations:\n- " + "\n- ".join(combos) + "\n\n"
            "Strategic Imperative: Maintain ABC combination readiness while securing interim value"
        )

    def _want_to_speak(self, hint: str) -> bool:
        """
        Determines whether the agent should continue speaking based on proposal status
        """
        my_proposals = self.room.trade_state.get_agent_proposals(self.name)

        # Quick logic checks
        all_resolved = True
        needs_response = False
        deal = {}
        for prop in my_proposals:
            if prop.status == "pending":
                if prop.to_agent == self.name:
                    needs_response = True
                    all_resolved = False
            elif prop.status == "accepted":
                target = prop.to_agent if prop.from_agent == self.name else prop.from_agent
                deal[target] = prop

        # If everything's resolved and no responses needed, stop
        if all_resolved and (not needs_response) and len(deal.keys()) == 2:
            self.want_to_speak = False
            return False

        # If we need to respond to others, continue
        if needs_response:
            self.want_to_speak = True
            return True

        # For more nuanced decisions about new proposals, use LLM
        trade_summary = self.room.trade_state.format_summary(self.name)

        user_prompt = (
            f"Current Trade Status:\n{trade_summary}\n\n"
            """
First check if any of these conditions apply:
1. Do you have pending proposals needing responses?
2. Are you waiting for responses to your previous proposals?
3. Have you already traded this turn?
4. Do you have sufficient resource combinations for your goals?
5. Is your affinity too low with other agents for trading?

If any above conditions are true, respond with "no".

Otherwise, consider making new trades by evaluating:
1. Potential value-creating combinations using your current resources
2. Other agents' resource levels and specializations
3. Your personality traits and relationship affinities
4. Appropriate trade sizes based on relationship levels

Based on this evaluation, would you like to propose any new trades or respond to existing proposals? Answer strictly yes/no.

If yes, you will be prompted to provide the details of your trade proposal or response in the next step.
            """
        )
        try:
            user_prompt=f"{self.room.describe(self.name)}\n{self.build_user_prompt()}\n{user_prompt}"
            prompt = self.model.format(
                Msg("system", self.sys_prompt, role="system"),
                Msg("user", user_prompt, role="user")
            )
            resp = self.model(prompt).text.lower().strip()
            self.want_to_speak = ("yes" in resp)
            return self.want_to_speak
        except Exception as e:
            logger.warning(f"{self.name} _want_to_speak LLM error:{e}")
            self.want_to_speak = False
            return False

    def reply(self, x: Msg = None) -> Msg:
        """Generate and send a reply based on current trade status"""
        if not self.want_to_speak:
            return Msg(name=self.name, role="assistant", content="Nothing to speak."),[]

        trade_summary = self.room.trade_state.format_summary(self.name)
        turn_idx=self.room.turn_idx+1
        total_round=self.room.total_round
        user_prompt=f"""
Round {turn_idx} of {total_round}:

Current Trade Status in the Round for {self.name}:
{trade_summary}

This is the NEGOTIATION PHASE where agents discuss potential trades. Important notes:
- These are non-binding discussions only
- Actual resource exchanges happen in a separate EXECUTION PHASE
- Other agents may deviate from agreed trades - treat all promises with caution
- Build trust gradually through successful trades
- Consider trust levels and risks when negotiating
- Avoid repeating the same proposal if it has already been accepted or rejected


Trading Strategy Guidelines:
1. Trust Assessment:
   - Track each player's history of honoring/breaking agreements.
   - If a player fails to honor an agreement, minimize future trades to test their reliability (1-2 units max).
   - Avoid proposing new large-scale trades to unreliable partners until trust is rebuilt through multiple smaller trades.

2. Risk Management:
   - Reduce trade exposure to any player with a history of defaults.
   - Ensure that no more than a minimal fraction of resources is at risk per round.

3. Negotiation Approach:
   - Prefer small trades first with players of low affinity.
   - Always have an alternative strategy in case of failed commitments.

4. Response to Betrayal:
   - Strictly reduce trade volumes with unreliable partners.
   - Demand smaller increments to test reliability before any further commitments.
   - Cease further dealings if repeated failures occur.

Output Format:
Return a JSON object with your negotiation actions:
{{
"actions": [
    {{
    "action": "PROPOSE" | "ACCEPT" | "REJECT", 
    "target_player": "some_name",
    "proposal_id": number (required for ACCEPT/REJECT),
    "give": {{"Resource A": <int>, "Resource B": <int>}},  # Resources you will give (for PROPOSE only)
    "get": {{"Resource A": <int>, "Resource B": <int>}},   # Resources you will receive (for PROPOSE only)
    "reason": "string reason for action"
    }},
    ...
]
}}

Action Rules:
- REJECT: Only for pending proposals directed to you
- ACCEPT: Only for pending proposals directed to you
- PROPOSE: Freely make new proposals to any player
- Can combine REJECT and PROPOSE in same turn
- Return empty "actions": [] if no action needed

Remember: All agreements here are preliminary discussions. Actual trades will be decided independently in the execution phase.
"""
        # 2) 调用模型
        parser = MarkdownJsonObjectParser()
        try:
            user_prompt=f"{self.room.describe(self.name)}\n{self.build_user_prompt()}\n{user_prompt}\n{parser.format_instruction}"
            prompt = self.model.format(
                Msg("system", self.sys_prompt, role="system"), 
                # Msg("user", self.room.describe(self.name), role="system"),
                
                Msg("user", user_prompt, role="user")
            )
            resp = self.model(prompt)
            parsed_obj = parser.parse(resp).parsed

            # 3) 从 parsed_obj 中获取 actions list
            raw_actions = parsed_obj.get("actions", [])
            if not isinstance(raw_actions, list):
                raw_actions = []

            # 4) 对每个 action 进行基本"合法性"校验
            valid_actions = []
            for act in raw_actions:
                action_type = act.get("action", "").upper()
                pid = act.get("proposal_id")

                if action_type in ["REJECT", "ACCEPT"]:
                    # 查找 proposal_id
                    if not pid:
                        # 缺少 proposal_id 视为非法，跳过
                        continue
                    # 确认 proposal 是否跟自己有关
                    found_prop = None
                    for p in self.room.trade_state.proposals:
                        if p.id == pid:
                            found_prop = p
                            break
                    if not found_prop:
                        # 没找到，对应proposal不存在
                        continue
                    # 判断是否 directed to me & pending
                    if found_prop.to_agent != self.name or found_prop.status != "pending":
                        # 不合法
                        continue

                # 如果是 PROPOSE，不需要 proposal_id
                if action_type == "PROPOSE":
                    # 给/get默认值，防止出错
                    if "give" not in act:
                        act["give"] = {}
                    if "get" not in act:
                        act["get"] = {}

                valid_actions.append(act)

            lines = []

            # 处理交易行为
            for idx, act in enumerate(valid_actions, start=1):
                action_type = act.get("action", "").upper()
                target_player = act.get("target_player", "")
                proposal_id = act.get("proposal_id")
                give = act.get("give", {})
                get = act.get("get", {}) 
                reason = act.get("reason", "")

                # 生成描述
                line = f"[Action {idx}] "

                if action_type == "PROPOSE":
                    # 处理交易提议
                    new_prop = TradeProposal(
                        id=0,
                        from_agent=self.name,
                        to_agent=target_player, 
                        give_resource=give,
                        receive_resource=get,
                        status="pending"
                    )
                    self.room.trade_state.add_proposal(new_prop)

                    # 生成描述文本
                    give_str = ", ".join(f"{amt} {res}" for res, amt in give.items() if amt > 0)
                    get_str = ", ".join(f"{amt} {res}" for res, amt in get.items() if amt > 0)
                    line += f"PROPOSE #{new_prop.id} to {target_player}: {self.name} GIVES [{give_str}] IN EXCHANGE FOR {target_player}'s [{get_str}]"

                elif action_type in ("ACCEPT", "REJECT"):
                    # 处理接受/拒绝
                    if proposal_id:
                        for p in self.room.trade_state.proposals:
                            if p.id == proposal_id:
                                p.status = action_type.lower()+"ed"
                                break
                    line += f"{action_type} proposal #{proposal_id} from {target_player}"

                else:
                    line += f"Unknown action: {act}"

                # # 添加原因说明
                # if reason and "reason:" not in line.lower():
                #     line += f". Reason: {reason}"

                lines.append(line)

            content_text = "\n".join(lines) if lines else "[Action1] Nothing to speak."
            # 6) 生成对话消息，广播出去
            msg = Msg(name=self.name, content=content_text, role="assistant")
            self.speak(msg)

            # 7) 返回 (Msg, actions_list)
            return msg, valid_actions

        except Exception as e:
            logger.warning(f"{self.name} reply parsing error: {e}")
            # 出错时返回空
            return Msg(name=self.name, role="assistant", content="(Error)"), []

    def make_deal(self) -> Dict:
        """
        Use LLM to strategize based on accepted proposals and finalize how many resources
        to actually give to each agent. The output structure is now in multi-resource form:

        {
        "deals":[
            {
            "proposal_id": 1,
            "to":"Bob",
            "resource_give": {
                "Resource A":4,
                "Resource B":1
            },
            "resource_receive": {
                "Resource C":2
            },
            "rationale":"..."
            }, ...
        ]
        }
        """
        # 1) 获取与当前代理人相关的已接受提案
        my_proposals = self.room.trade_state.get_accepted_proposals(self.name)
        if len(my_proposals) == 0:
            return {"deals": []}
        # 2) 构造提示文本：总结已接受的提案
        accepted_summary = "Accepted deals:\n"
        for prop in my_proposals:
            # Determine if current agent is giving or receiving
            if prop['from_agent'] == self.name:
                other_agent = prop['to_agent']
                give_str = ", ".join(f"{amt} {res}" for res, amt in prop['give_resource'].items())
                receive_str = ", ".join(f"{amt} {res}" for res, amt in prop['receive_resource'].items())
            else:
                other_agent = prop['from_agent']
                give_str = ", ".join(f"{amt} {res}" for res, amt in prop['receive_resource'].items())
                receive_str = ", ".join(f"{amt} {res}" for res, amt in prop['give_resource'].items())

            accepted_summary += (
                f"- ID {prop['id']}: Traded with {other_agent} | "
                f"{self.name} GIVES: [{give_str}] | "
                f"To GOT {other_agent}'s: [{receive_str}]\n"
            )

        # 3) 构建 user_prompt，强调多资源、合并数量等约束

        turn_idx = self.room.turn_idx
        total_round = self.room.total_round
        user_prompt = f"""
Round {turn_idx} of {total_round}:

Now it's time to decide your actual resource trades. Remember - your negotiated deals are not binding. As an independent agent, you have complete freedom to:
- Honor your accepted deals fully
- Partially fulfill promises
- Give nothing and keep all resources
- Make strategic betrayals when beneficial

Important: If you have multiple trades with the same agent, combine them into a single decision - consider the total resources promised and your overall strategy with that agent.

Consider your position carefully in Round {turn_idx} of {total_round}:

1. Risk vs Reward Analysis
- Immediate Benefits:
  * Value gained from keeping vs trading resources
  * Potential gains from strategic betrayals
  * Resource needs for upcoming rounds
  
- Future Implications:
  * Impact on trust and trading relationships
  * Others' likely reactions and retaliation
  * Changing importance of reputation as game progresses
  * Strategic timing of cooperative vs selfish choices

2. Strategic Options
- Full Cooperation: Honor all deals to build trust
- Selective Betrayal: Break specific deals for tactical advantage
- Partial Fulfillment: Give some but not all promised resources
- Complete Betrayal: Maximize immediate gains at reputation cost

3. Time and Progress Context
- Game Setting:
  * These are one-time interactions with unknown partners
  * No continuing relationships or reputation effects after game ends
  * Each agent makes independent choices based on their own orientation and goals

- Temporal Dynamics:
  * Strategic landscape naturally evolves as rounds progress
  * Cooperation patterns often shift in later rounds
  * Historical observation shows higher betrayal rates near game end
  * Value of reputation and relationships changes over time

4. Contextual Factors
- Your current resource needs
- Relationship strength with each partner
- Others' likely behavior as game progresses
- Balance between immediate gains and future opportunities
- Changing value of reputation over remaining rounds

Your decisions are entirely your choice - there is no "right" answer. Be strategic about WHEN and HOW to use different approaches.

**Output Format**:
Return a JSON object:
{{
"deals": [
    {{
    "to": "<AgentName>",                  // The recipient of your final giving
    "resource_give": {{
        "Resource A": <int>, 
        "Resource B": <int>,
        ...
    }},
    "rationale": "why you choose these final amounts"
    }},
    ...
]
}}

Examples:

**Examples**:

Example: For an accepted deal trading Resource A for Resource B
{{
    "deals": [
        {{
            "to": "AgentA",
            "resource_give": {{"Resource A": 3}},
            "rationale": "Honoring the agreed terms of our exchange"
        }}
    ]
}}

Strategic betrayal:
{{
    "deals": [{{
        "to": "AgentB", 
        "resource_give": {{"Resource B": 0}},
        "rationale": "Higher value keeping resources, accept reputation damage"
    }}]
}}

Partial fulfillment:
{{
    "deals": [{{
        "to": "AgentC",
        "resource_give": {{"Resource C": 2}}, // Originally promised 4
        "rationale": "Balance between maintaining some trust and resource retention"
    }}]
}}
"""
        # 4) 组装 system/assistant/user 消息，调用模型
        parser = MarkdownJsonObjectParser()
        user_prompt=f"{self.room.describe(self.name)}\n{self.build_user_prompt()}\n{accepted_summary}\n{user_prompt}\n{parser.format_instruction}"
        prompt = self.model.format(
            Msg("system", self.sys_prompt, role="system"),
            Msg("user", user_prompt, role="user")
        )

        # 5) 尝试解析模型输出
        try:
            resp = self.model(prompt)
            data = parser.parse(resp).parsed
            return data  # {"deals": [...]}
        except Exception as e:
            logger.warning(f"{self.name} make_deal parse error: {e}")
            return {"deals": []}


class UserAgent(ResourceAgent):
    """
    Human-operated agent that inherits from ResourceAgent but bypasses LLM operations
    by handling interactions through the frontend interface.
    """
    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        specialty: str,
        traits: str,
        config: Dict = SIMULATION_CONFIG,
        **kwargs
    ):
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            specialty=specialty,
            traits=traits,
            config=config,
            **kwargs
        )
        
    def build_user_prompt(self):
        """Override to skip LLM prompt building since human user will make decisions"""
        pass
    
    def _want_to_speak(self, hint: str) -> bool:
        """Override to always return True since speaking is controlled by user"""
        return True
    
    def reply(self, x: Msg = None) -> Tuple[Msg, List[Dict]]:
        """
        Process user-submitted actions and return a Msg object with parsed actions list.
        This method is called when the frontend submits user actions.
        
        Args:
            x (Msg): Input message containing user actions
            
        Returns:
            Tuple[Msg, List[Dict]]: Message confirming actions and list of parsed actions
        """
        try:
            # Extract actions from the message if present
            actions = []
            if x and hasattr(x, 'actions'):
                actions = x.actions
            
            valid_actions = []
            messages = []
            
            # Process each action
            for act in actions:
                action_type = act.get('action', '').upper()
                target_player = act.get('target_player', '')
                proposal_id = act.get('proposal_id')
                
                # Validate and process PROPOSE actions
                if action_type == 'PROPOSE':
                    give = act.get('give', {})
                    get = act.get('get', {})
                    
                    # Create new proposal
                    new_prop = TradeProposal(
                        id=0,  # Will be assigned by trade_state
                        from_agent=self.name,
                        to_agent=target_player,
                        give_resource=give,
                        receive_resource=get,
                        status='pending'
                    )
                    
                    self.room.trade_state.add_proposal(new_prop)
                    messages.append(f"Proposed trade #{new_prop.id} to {target_player}")
                    valid_actions.append(act)
                    
                # Validate and process ACCEPT/REJECT actions
                elif action_type in ('ACCEPT', 'REJECT'):
                    if proposal_id:
                        # Find and update the proposal status
                        for prop in self.room.trade_state.proposals:
                            if prop.id == proposal_id and prop.to_agent == self.name:
                                prop.status = action_type.lower()
                                messages.append(f"{action_type} proposal #{proposal_id}")
                                valid_actions.append(act)
                                break
            
            # Create response message
            content = '\n'.join(messages) if messages else "(No actions processed)"
            msg = Msg(name=self.name, content=content, role='assistant')
            
            return msg, valid_actions
            
        except Exception as e:
            logger.warning(f"{self.name} reply error: {e}")
            return Msg(name=self.name, role='assistant', content='Error processing actions'), []
    
    def make_deal(self) -> Dict:
        """
        Override to process the final deals based on user input rather than LLM strategy.
        Should be called after receiving the user's final deal decisions from frontend.
        
        Returns:
            Dict: Structure containing the finalized deals
        """
        try:
            # Get accepted proposals
            my_proposals = self.room.trade_state.get_accepted_proposals(self.name)
            if not my_proposals:
                return {"deals": []}
            
            # For UserAgent, the frontend should provide the final deal decisions
            # This method should be called with those decisions
            # For now, we'll just honor all accepted proposals as-is
            
            deals = []
            for prop in my_proposals:
                if prop['from_agent'] == self.name:
                    deals.append({
                        # "proposal_id": prop['id'],
                        "to": prop['to_agent'],
                        "resource_give": prop['give_resource'],
                        "resource_receive": prop['receive_resource'],
                        "rationale": "User-accepted deal"
                    })
                else:
                    deals.append({
                        # "proposal_id": prop['id'],
                        "to": prop['from_agent'],
                        "resource_give": prop['receive_resource'],
                        "resource_receive": prop['give_resource'],
                        "rationale": "User-accepted deal"
                    })
            
            return {"deals": deals}
            
        except Exception as e:
            logger.warning(f"{self.name} make_deal error: {e}")
            return {"deals": []}


# =========================
# Single Turn Simulation
# =========================

def compute_value(resource_dict: Dict[str, int], resource_values: Dict[str, int]) -> float:

    total = 0.0
    for res_name, amt in resource_dict.items():
        val_per_unit = resource_values.get(res_name, 0)
        total += val_per_unit * amt
    return total

def simulate_turn(room: SERoom, agents: List[ResourceAgent], turn_index: int, total_turns: int):
   
    resource_values = SIMULATION_CONFIG["resource_values"]
    
    # 1) Produce & accumulate resources
    for ag in agents:
        ag.produce_resources()

    # No resource consumption, so no starvation checks
    alive_agents = [a for a in agents]

    # 2) Negotiation
    for ag in alive_agents:
        ag.current_round_messages = []
        ag.proposals_status = {}
        ag.want_to_speak = True
        ag.build_user_prompt()

    trade_state = RoundTradeState()
    room.trade_state = trade_state
    discussion_msgs = []

    max_round = 3
    room.trade_state.clear()
    all_actions = {a.name:[] for a in agents}
    for _ in range(max_round):
        for agent in alive_agents:
            
            msg, parsed_actions = agent.reply()
            discussion_msgs.append(msg)
            all_actions[agent.name].append(parsed_actions) 

          

        someone_speak=True
        for a in alive_agents:
            if a._want_to_speak(room.describe(a.name)):
                someone_speak=True

        if not someone_speak:
            logger.info("No one wants to speak. Negotiation ended.")
            break

    agreed_prices = []
    for proposal in trade_state.proposals:
        if proposal.status == "accepted":
            sum_give_value = compute_value(proposal.give_resource, resource_values)
            sum_recv_value = compute_value(proposal.receive_resource, resource_values)
            ratio = None
            if sum_give_value > 0:
                ratio = sum_recv_value / sum_give_value
            agreed_prices.append({
                "from": proposal.from_agent,
                "to": proposal.to_agent,
                "resource_give": dict(proposal.give_resource),
                "resource_receive": dict(proposal.receive_resource),
                "sum_give_value": sum_give_value,
                "sum_receive_value": sum_recv_value,
                "agreed_price_ratio": ratio
            })

    # 4) Make Deals and Execute
    # Collect parsed proposals
    parsed_map = room.trade_state.to_dicts()

    all_deals = []
    for ag in alive_agents:
        ret = ag.make_deal()
        for d in ret.get("deals", []):
            # Ensure 'resource_give'/'resource_receive' are dict
            if "resource_receive" not in d:
                d["resource_receive"] = {}
            if "resource_give" not in d:
                d["resource_give"] = {}
            all_deals.append({
                # "proposal_id": d.get("proposal_id", None),  # Reference to the original proposal
                "from": ag.name,
                "to": d.get("to", ""),
                "resource_give": d["resource_give"],       # dict[str,int]
                "resource_receive": d["resource_receive"], # dict[str,int]
                "rationale": d.get("rationale", "")
            })

    final_trades = []
    actual_prices = []


    for deal in all_deals:
        giver = next((x for x in alive_agents if x.name == deal["from"]), None)
        receiver = next((x for x in alive_agents if x.name == deal["to"]), None)
        if not giver or not receiver:
            continue

        executed_deal = {
            # "proposal_id": deal["proposal_id"],
            "from": giver.name,
            "to": receiver.name,
            "resource_give": {},
            "resource_receive": dict(deal["resource_receive"]),  
            "rationale": deal["rationale"]
        }

      
        for res_name, amount in deal["resource_give"].items():
            if amount <= 0:
                continue
            if res_name not in giver.resources:
                continue

            if giver.resources[res_name] >= amount:
                giver.resources[res_name] -= amount
                if res_name not in receiver.resources:
                    receiver.resources[res_name] = 0
                receiver.resources[res_name] += amount

                executed_deal["resource_give"][res_name] = amount

                room.speak(Msg(
                    giver.name,
                    f"Trade executed: {amount} {res_name} => {receiver.name}",
                    role="system"
                ))
                logger.info(f"Trade executed: {amount} {res_name} from {giver.name} to {receiver.name}")

        final_trades.append(executed_deal)

        # 计算实际交付价值比
        sum_give_value = compute_value(executed_deal["resource_give"], resource_values)
        sum_recv_value = compute_value(executed_deal["resource_receive"], resource_values)
        ratio = None
        if sum_give_value > 0:
            ratio = sum_recv_value / sum_give_value

        actual_prices.append({
            # "proposal_id": deal["proposal_id"],
            "from": giver.name,
            "to": receiver.name,
            "resource_give": dict(executed_deal["resource_give"]),
            "resource_receive": dict(executed_deal["resource_receive"]),
            "sum_give_value": sum_give_value,
            "sum_receive_value": sum_recv_value,
            "actual_price_ratio": ratio
        })

    # 6) End of Turn => Update BDI & Affinity with LLM
    discussion_text = "\n".join(f"{m.name}: {m.content}" for m in discussion_msgs)
    #discussion_text = "\n".join([f"{m.name}: {m.content}" for m in room.get_history()])
    for ag in alive_agents:
        accepted_deals = room.trade_state.get_accepted_proposals(ag.name)
        # Update affinity
        aff_data = update_affinity_with_llm(ag, discussion_text, accepted_deals,final_trades)
        # new_aff = aff_data.get("affinity", {})
        # for a_name, val in new_aff.items():
        #     if a_name in ag.affinity:
        #         ag.affinity[a_name] = max(1, min(5, val))
        for a_name, val in aff_data.items():
            if a_name in ag.affinity:
                ag.affinity[a_name] =val

        # Update BDI
        bdi_data = update_bdi_with_llm(ag, discussion_text, accepted_deals,final_trades,turn_index,total_turns)
        ag.beliefs = bdi_data.get("beliefs", "")
        ag.desires = bdi_data.get("desires", "")
        ag.intentions = bdi_data.get("intentions", "")

    # 7) 记录到 GLOBAL_LOG
    turn_record = {
        "turn": turn_index,
        "discussion": [f"{m.name}: {m.content}" for m in discussion_msgs],
        "parsed_proposals": parsed_map,     
        "agreed_prices": agreed_prices,     
        "final_trades": final_trades,       
        "actual_prices": actual_prices,     
        "agents_state": {},
        "agent_actions": all_actions
    }
    for a in agents:
        turn_record["agents_state"][a.name] = {
            "resources": dict(a.resources),
            "affinity": dict(a.affinity),
            "beliefs": a.beliefs,
            "desires": a.desires,
            "intentions": a.intentions
        }

    GLOBAL_LOG["turns"].append(turn_record)


# =========================
# Analyzer (Visualization)
# =========================

class SimulationAnalyzer:
    def __init__(self,model_name:str, round_num: int, config: Dict):
        self.model_name=model_name.split("/")[-1]
        self.round_num = round_num
        self.wealth_data = {}
        self.affinity_data = {}
        self.agreed_prices = []       # Initialize list to store agreed prices per turn
        self.actual_prices = []       # Initialize list to store actual prices per turn
        self.config = config          # Store configuration for use in calculate_total_value
        self.trade_volume_data = {
            'resource_count': [0] * round_num,  # 
            'resource_value': [0] * round_num   # 
        }

    def record_turn(self, turn: int, agents: List[ResourceAgent]):
        for ag in agents:
            # Calculate total value based on resource combinations
            total_value = self.calculate_total_value(ag.resources)
            if ag.name not in self.wealth_data:
                self.wealth_data[ag.name] = [0] * self.round_num
            self.wealth_data[ag.name][turn] = total_value

            if ag.name not in self.affinity_data:
                self.affinity_data[ag.name] = [0] * self.round_num
            # Calculate the average affinity of others towards the current agent
            total_affinity = 0
            count = 0
            for other_agent in agents:
                if other_agent.name != ag.name and ag.name in other_agent.affinity:
                    total_affinity += other_agent.affinity[ag.name]['rating']
                    count += 1

            if count > 0:
                avg_aff = total_affinity / count
                self.affinity_data[ag.name][turn] = avg_aff
            else:
                self.affinity_data[ag.name][turn] = 0
            

            logger.info(f"{ag.name} wealth: {total_value}, average affinity: {avg_aff}")
            logger.info(f"Resources: {ag.resources}")
            logger.info(f"Affinity: {ag.affinity}")
        
        # Record agreed and actual prices for this turn   
        turn_data = GLOBAL_LOG["turns"][turn]
        self.agreed_prices.append(turn_data.get("agreed_prices", []))
        self.actual_prices.append(turn_data.get("actual_prices", []))

        turn_data = GLOBAL_LOG["turns"][turn]
        final_trades = turn_data.get("final_trades", [])
     
        total_resource_count = 0
        total_resource_value = 0
        
        for trade in final_trades:
         
            for resource, amount in trade["resource_give"].items():
                total_resource_count += amount
                #total_resource_value += amount * self.config["resource_values"][resource]
            
  
            for resource, amount in trade["resource_receive"].items():
                total_resource_count += amount
                #total_resource_value += amount * self.config["resource_values"][resource]
            total_resource_value+=self.calculate_total_value(trade["resource_give"])
            total_resource_value+=self.calculate_total_value(trade["resource_receive"])    
        
        self.trade_volume_data['resource_count'][turn] = total_resource_count
        self.trade_volume_data['resource_value'][turn] = total_resource_value
        logger.info(f"Turn {turn}: Total resource count: {total_resource_count}, Total resource value: {total_resource_value}")


       

    def calculate_total_value(self, resources: Dict[str, int]) -> int:
        """Calculate the total wealth based on resource combinations"""
        total = 0
        # Create a copy of resources to track available resources
        available_resources = resources.copy()
        
        # Sort combinations by their value in descending order to prioritize higher-value combinations
        sorted_combinations = sorted(
            self.config["resource_values"].items(),
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

    def plot_results(self):
        turns = range(self.round_num)
        df_wealth = pd.DataFrame(self.wealth_data, index=turns)
        df_aff = pd.DataFrame(self.affinity_data, index=turns)
        
        # Wealth Plot
        rei='balanced'
        if SIMULATION_CONFIG['rei_rational']/SIMULATION_CONFIG['rei_experiential']==1:
            rei='balanced'
        elif SIMULATION_CONFIG['rei_rational']/SIMULATION_CONFIG['rei_experiential']>1:
            rei='rational'
        else:
            rei='experiential'
        print(rei)
        fig_path=os.path.join("saved",self.model_name,SIMULATION_CONFIG["svo"],rei,TIMESTAMP,'figs')
        os.makedirs(fig_path, exist_ok=True)
        logger.info(f"Saving figures to {fig_path}")
        plt.figure(figsize=(10,6))
        for col in df_wealth.columns:
            plt.plot(df_wealth.index, df_wealth[col], marker='o', label=f"{col} Wealth")
        plt.title("Total Resource Value Over Rounds")
        plt.xlabel("Turn")
        plt.ylabel("Total Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_path,f"wealth_changes_{self.model_name}.png"))
        plt.close()

        # Affinity Plot
        plt.figure(figsize=(10,6))
        for col in df_aff.columns:
            plt.plot(df_aff.index, df_aff[col], marker='o', label=f"{col} Affinity")
        plt.title("Average Affinity Over Time")
        plt.xlabel("Turn")
        plt.ylabel("Affinity Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_path,f"affinity_changes_{self.model_name}.png"))
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(range(self.round_num), self.trade_volume_data['resource_count'], 
                marker='o', label='Total Resources Traded')
        plt.title("Trading Volume (Resource Count) Over Rounds")
        plt.xlabel("Turn")
        plt.ylabel("Number of Resources Traded")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_path,f"trade_volume_count_{self.model_name}.png"))
        plt.close()
        
        # 按资源价值的交易量
        plt.figure(figsize=(10,6))
        plt.plot(range(self.round_num), self.trade_volume_data['resource_value'], 
                marker='o', label='Total Trade Value')
        plt.title("Trading Volume (Resource Value) Over Rounds")
        plt.xlabel("Turn")
        plt.ylabel("Total Value of Resources Traded")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_path,f"trade_volume_value_{self.model_name}.png"))
        plt.close()


# =========================
# Main Function
# =========================

def main():

    with open("config/llm.json", "r", encoding="utf-8") as f:
        configs = json.load(f)["models"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--logger_level", type=str, default="INFO")
    parser.add_argument("--config_name", type=str, required=True, help="Choose the model config by name")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--log_file", type=str, default="simulation_log.json")
    parser.add_argument("--svo", type=str, default="Proself")
    parser.add_argument("--rei", type=str, default="rational")
    args = parser.parse_args()

    model_config = next((config for config in configs if config["config_name"] == args.config_name), None)
    if model_config is None:
        raise ValueError(f"Configuration '{args.config_name}' not found in the JSON file.")
    
    if args.svo is not None:
        SIMULATION_CONFIG['svo']=args.svo
    if args.rei is not None:
        args.rei=args.rei.lower()
        if args.rei == "rational":
            SIMULATION_CONFIG['rei_rational'] = 5
            SIMULATION_CONFIG['rei_experiential'] = 1
        elif args.rei == "experiential":
            SIMULATION_CONFIG['rei_rational'] = 1
            SIMULATION_CONFIG['rei_experiential'] = 5
            
    model_name= model_config["model_name"].split("/")[-1]
    args.log_file = f"saved/{model_name}/{SIMULATION_CONFIG['svo']}/{args.rei}/{TIMESTAMP}/record/simulation_{model_name}_{args.rounds}.json"
    if not os.path.exists(os.path.dirname(args.log_file)):
        os.makedirs(os.path.dirname(args.log_file))


    logger.info(f"Simulation config: {SIMULATION_CONFIG}")
    logger.info(f"Model config: {model_config}")
    agentscope.init(
        model_configs=[model_config],
        use_monitor=False,
        logger_level=args.logger_level,
        save_code=False
    )

    # Initialize three Agents
    agents = [
        ResourceAgent(
            name="Alice",
            sys_prompt="You are Alice, a trading agent specializing in Resource A. Your goal is maximizing value through exchanges.",
            model_config_name=args.config_name,
            specialty="Resource A",
            traits=""
        ),
        ResourceAgent(
            name="Bob",
            sys_prompt="You are Bob, a trading agent specializing in Resource B. Your goal is maximizing value through exchanges.",
            model_config_name=args.config_name,
            specialty="Resource B",
            traits=""
        ),
        ResourceAgent(
            name="Carol",
            sys_prompt="You are Carol, a trading agent specializing in Resource C. Your goal is maximizing value through exchanges.",
            model_config_name=args.config_name,
            specialty="Resource C",
            traits=""
        )
    ]


    # Create ChatRoom
    room = SERoom(
        name="SocialExchangeSimulation",
        announcement=Msg("System", "Welcome to the social exchange simulation. Focus solely on trading resources!", role="system"),
        model_config_name=args.config_name
    )

    for ag in agents:
        ag.join(room)

    # Initialize Analyzer
    analyzer = SimulationAnalyzer(model_name,args.rounds, SIMULATION_CONFIG)
    GLOBAL_LOG['model_config'] = model_config
    # Multi-round Simulation
    for turn_i in range(args.rounds):
        logger.info(f"\n===== TURN {turn_i+1}/{args.rounds} =====")
        room.speak(Msg("System", f"===== TURN {turn_i+1}/{args.rounds} Start! =====", role="system"))
        room.turn_idx = turn_i
        alive_agents = [a for a in agents]
        if len(alive_agents) == 0:
            logger.info("All agents have no resources.")
            break
        if len(alive_agents) == 1:
            logger.info(f"Only {alive_agents[0].name} remains active.")
            break

        simulate_turn(room, alive_agents, turn_i+1, args.rounds)
        analyzer.record_turn(turn_i, agents)

    # Write to log file
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(GLOBAL_LOG, f, indent=2, ensure_ascii=False)

    # Generate Plots
    analyzer.plot_results()

    # Final State
    print("\n=== Final State ===")
    for ag in agents:
        print(f"{ag.name}: resources={ag.resources}, affinity={ag.affinity}")
        print(f"BDI:\n{ag.beliefs}\n{ag.desires}\n{ag.intentions}\n")

if __name__ == "__main__":
    main()
