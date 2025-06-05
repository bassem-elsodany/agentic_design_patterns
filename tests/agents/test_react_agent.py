"""
test_react_agent.py

This test demonstrates how to use the ReactAgent for step-by-step reasoning and tool use. 
It is designed for beginners to see how an agent can chain thoughts and actions to solve a multi-step problem using tools.

Key Concepts:
- ReAct pattern: Alternates between reasoning (thoughts) and actions (tool calls).
- AgentFactory: A helper to create different types of agents for testing.

Example usage:
    pytest tests/agents/test_react_agent.py
"""

import pytest
from src.agents.helpers.agent_factory import AgentFactory
from src.agents.helpers.tool import tool
from utils.logging import get_logger
from utils.demo_tools import get_weather, calculate_area, recommend_food

logger = get_logger("TestReactAgent")

SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools>.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}}
</tool_call>

Here are the available tools / actions:

<tools>
{tools}
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{{"name": "get_current_weather","arguments": {{"location": "Madrid", "unit": "celsius"}}, "id": 0}}</tool_call>

You will be called again with this:

<observation>{{0: {{"temperature": 25, "unit": "celsius"}}}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""

def test_react_agent_chain_of_thought():
    tools = [get_weather, calculate_area, recommend_food]
    user_prompt = (
        "I'm planning a picnic in Paris. What is the weather like, and if it's good, "
        "what is the area of a circle with radius 2 meters for our picnic blanket? "
        "Also, what foods would you recommend for a picnic in Paris? "
        "Please show your reasoning and tool use step by step."
    )

    agent = AgentFactory.create(
        "react",
        tools=tools,
        model="llama-3.3-70b-versatile",
        system_prompt=SYSTEM_PROMPT
    )
    logger.debug(f"Final system prompt used:\n{agent.system_prompt}")
    output = agent.run(user_message=user_prompt, max_rounds=3)
    assert isinstance(output, str)
    logger.info(f"Chain-of-Thought ReactAgent output: {output}") 