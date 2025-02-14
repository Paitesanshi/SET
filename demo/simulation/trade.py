# simulation/trade.py

from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import json
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
        #self.id_counter = 1  # 若希望每轮清空ID计数可以取消注释
    
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
        """Get accepted proposals for an agent"""
        return [
            p.to_dict() for p in self.proposals 
            if (p.to_agent == agent_name or p.from_agent == agent_name) 
            and p.status == "accepted"
        ]