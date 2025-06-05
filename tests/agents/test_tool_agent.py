"""
test_tool_agent.py

This test demonstrates how to use the ToolAgent for function/tool calling with LLMs. 
It is designed for beginners to see how an agent can use tools to answer questions and follow a reasoning process.

Key Concepts:
- ToolAgent: An agent that can call functions to help answer questions.
- AgentFactory: A helper to create different types of agents for testing.

Example usage:
    pytest tests/agents/test_tool_agent.py
"""

import pytest
from src.agents.helpers.agent_factory import AgentFactory
from src.agents.helpers.tool import tool
from utils.logging import get_logger
from utils.demo_tools import get_planet_distance

logger = get_logger("TestToolAgent")

def test_tool_agent_basic():

    TOOL_SYSTEM_PROMPT = """
You are an AI assistant capable of calling functions to help answer user questions. The available functions are described within <tools></tools> tags below.
To solve the user's query, you may call one or more functions. Do not guess argument values—always use the types as described in the function signatures, formatted as Python dicts.
For each function call, respond with a JSON object containing the function name, arguments, and a unique id, wrapped in <tool_call></tool_call> tags, like this:

<tool_call>
{{"name": <function-name>, "arguments": <args-dict>, "id": <unique-id>}}
</tool_call>

Available functions:
<tools>
{{tools}}
</tools>

If the user's question cannot be answered with the provided functions, reply directly, enclosing your answer in <response></response> tags.
"""

    agent = AgentFactory.create(
        "tool",
        tools=[get_planet_distance],
        model="llama-3.3-70b-versatile",
        system_prompt=TOOL_SYSTEM_PROMPT
    )
    user_prompt = (
        "You have access to the following tool: get_planet_distance(planet: str) -> str. "
        "What is the distance from Earth to Mars? Please show your reasoning and tool use step by step, "
        "using the tool as needed."
    )
    output = agent.run(user_message=user_prompt)
    assert isinstance(output, str)
    logger.info(f"ToolAgent output: {output}") 