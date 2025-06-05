"""
test_agent.py

This test demonstrates how to use the multi-agent system for collaborative task planning. 
It is designed for beginners to see how to instantiate and run a multi-agent agent using the AgentFactory.

Key Concepts:
- Multi-agent: Agents that can work together and share context.
- AgentFactory: A helper to create different types of agents for testing.

Example usage:
    pytest tests/agents/test_agent.py
"""

import pytest
from src.agents.helpers.agent_factory import AgentFactory
from utils.logging import get_logger

logger = get_logger("TestMultiAgent")

def test_multiagent_agent_basic():
    agent = AgentFactory.create(
        "multiagent",
        name="TravelPlanner",
        backstory="You are an expert travel planner specializing in European destinations.",
        task_description="Plan a 3-day itinerary for Paris for a family with two kids.",
        task_expected_output="A detailed 3-day itinerary with activities suitable for children.",
        tools=None,
        llm="llama-3.3-70b-versatile"
    )
    output = agent.run()
    assert isinstance(output, str)
    logger.info(f"Multiagent Agent output: {output}") 