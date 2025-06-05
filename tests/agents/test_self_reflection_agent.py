"""
test_self_reflection_agent.py

This test demonstrates how to use the SelfReflectionAgent for iterative answer improvement. 
It is designed for beginners to see how an agent can generate, critique, and refine its own responses using LLMs.

Key Concepts:
- Self-reflection: The agent reviews and improves its own output.
- AgentFactory: A helper to create different types of agents for testing.

Example usage:
    pytest tests/agents/test_self_reflection_agent.py
"""

import pytest
from src.agents.helpers.agent_factory import AgentFactory
from utils.logging import get_logger

logger = get_logger("TestSelfReflectionAgent")

def test_self_reflection_agent_basic():
    agent = AgentFactory.create(
        "reflection",
        model="llama-3.3-70b-versatile",
        history_length=3
    )
    human_input = "Write a short story about a robot learning to paint."
    system_prompt = "You are Isaac Asimov, a master of science fiction storytelling."
    stop_token = "<SATISFIED>"
    self_reflection_prompt = (
        "You are a literary critic. Provide feedback on creativity, clarity, and emotional impact. "
        f"If the story is perfect, reply with {stop_token}."
    )

    result = agent.run(
        human_input=human_input,
        system_prompt=system_prompt,
        self_reflection_prompt=self_reflection_prompt,
        max_cycles=3,
        stop_token=stop_token,
        verbose=1
    )
    assert hasattr(result, "final_candidate")
    assert hasattr(result, "observations")
    logger.info(f"Final candidate: {result.final_candidate}")
    logger.info(f"Observations: {result.observations}") 