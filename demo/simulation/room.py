# simulation/room.py

from typing import List, Dict, Any
from agentscope.message import Msg
from simulation.trade import TradeProposal, RoundTradeState

class SERoom:
    """Simple Exchange Room"""
    def __init__(self, name: str, announcement: TradeProposal, model_config_name: str):
        self.name = name
        self.announcement = announcement
        self.model_config_name = model_config_name
        self.agents: List[Any] = []
        self.messages: List[Msg] = []
        self.trade_state = RoundTradeState()
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def speak(self, msg: Msg):
        self.messages.append(msg)
        # 可以在这里广播消息给所有代理人，视具体实现而定