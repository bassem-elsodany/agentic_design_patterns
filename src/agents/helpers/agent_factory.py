"""
agent_factory.py

This module provides a simple factory for creating different types of agentic agents. 
It is designed for beginners to easily instantiate various agent classes (reflection, tool, react, multiagent) by name.

Key Concepts:
- Factory pattern: A design pattern that provides a way to create objects without specifying the exact class.
- Agent types: Different agent classes for different reasoning and tool-use strategies.

Example usage:
    agent = AgentFactory.create('tool', tools=[...], model='llama-3.3-70b-versatile')
    output = agent.run(user_message='What is the weather?')
    print(output)
"""

from src.agents.self_reflection_agent import SelfReflectionAgent
from src.agents.tool_agent import ToolAgent
from src.agents.react_agent import ReactAgent
from src.agents.multi_agent import Agent

class AgentFactory:
    @staticmethod
    def create(agent_type: str, **kwargs):
        """
        Factory method to create agent instances by type.

        Args:
            agent_type (str): The type of agent to create (e.g., 'reflection', 'tool', 'react', 'multiagent').
            **kwargs: Arguments to pass to the agent constructor.

        Returns:
            An instance of the requested agent type.
        """
        agent_type = agent_type.lower()
        if agent_type == "reflection":
            return SelfReflectionAgent(**kwargs)
        elif agent_type == "tool":
            return ToolAgent(**kwargs)
        elif agent_type == "react":
            return ReactAgent(**kwargs)
        elif agent_type == "multiagent":
            return Agent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}") 