# simulation/agents.py

from typing import List, Dict, Any,Tuple
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonObjectParser
from simulation.trade import TradeProposal
from simulation.room import SERoom
from simulation.helpers import extract_proposals_with_llm, update_bdi_with_llm, update_affinity_with_llm
from agentscope.message import Msg

class ResourceAgent:
    """
    代理人类，自动通过 LLM 生成对话和操作
    """
    def __init__(self, name: str, sys_prompt: str, model_config_name: str, specialty: str, traits: str, config: Dict):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model_config_name = model_config_name
        self.specialty = specialty
        self.traits = traits
        self.config = config
        self.resources = dict(self.config["initial_resources"])
        self.affinity = {a: 3 for a in ["Alice", "Bob", "Carol"] if a != self.name}
        self.svo = self.config.get("svo", "Individualistic")
        self.rei_rational = self.config.get("rei_rational", 3)
        self.rei_experiential = self.config.get("rei_experiential", 3)
        self.bdi_reflection = ""
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}
        self.proposals_status = {}
        self.current_round_messages = []
        self.want_to_speak = True
        self.room = None  # 需要在 join() 时设置
    
    def join(self, room: SERoom):
        self.room = room
        room.add_agent(self)
    
    def produce_resources(self):
        """Produce resources based on specialization"""
        if self.specialty in self.resources:
            self.resources[self.specialty] += self.config["resources"][self.specialty]["production"]
        # No resource consumption, so no deduction
    
    def reply(self) -> (Msg, List[Dict]):
        """
        生成并发送回复消息，返回 Msg 对象和解析的动作列表
        """
        # 这里应集成 LLM 生成回复的逻辑
        # 简化示例，发送固定消息
        message = f"{self.name} 生成了一条消息。"
        msg = Msg(name=self.name, content=message, role="assistant")
        return msg, []
    



class UserAgent:
    """
    人类代理人，通过前端操作进行交互
    """
    def __init__(self, name: str, sys_prompt: str, model_config_name: str, specialty: str, traits: str, config: Dict):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model_config_name = model_config_name
        self.specialty = specialty
        self.traits = traits
        self.config = config
        self.resources = dict(self.config["initial_resources"])
        self.affinity = {a: 3 for a in ["Alice", "Bob", "Carol"] if a != self.name}
        self.svo = self.config.get("svo", "Individualistic")
        self.rei_rational = self.config.get("rei_rational", 3)
        self.rei_experiential = self.config.get("rei_experiential", 3)
        self.bdi_reflection = ""
        self.beliefs = {}
        self.desires = {}
        self.intentions = {}
        self.proposals_status = {}
        self.current_round_messages = []
        self.want_to_speak = True
        self.room = None  # 需要在 join() 时设置
    
    def join(self, room: SERoom):
        self.room = room
        room.add_agent(self)
    
    def produce_resources(self):
        """Produce resources based on specialization"""
        if self.specialty in self.resources:
            self.resources[self.specialty] += self.config["resources"][self.specialty]["production"]
        # No resource consumption, so no deduction
    
    def reply(self, actions: List[Dict]) -> Tuple[Msg, List[Dict]]:
        """
        处理用户提交的操作，并返回 Msg 对象和解析的动作列表
        """
        # 根据用户提交的 actions 进行处理
        # 简化示例，仅返回一个确认消息
        message = f"{self.name} 执行了用户提交的操作。"
        msg = Msg(name=self.name, content=message, role="assistant")
        return msg, actions