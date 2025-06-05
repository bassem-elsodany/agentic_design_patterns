"""
tool_agent.py

This module defines the ToolAgent class, which lets agents use external tools (functions) to answer user questions. 
It is designed for beginners to see how an agent can call functions, validate arguments, and combine tool results with LLM reasoning.

Key Concepts:
- ToolAgent: An agent that can call one or more tools to help answer questions.
- Tool call: The agent decides which function to use and with what arguments.
- System prompt: Guides the agent on how to use tools and format its responses.

Example usage:
    agent = ToolAgent(tools=[...], model="llama-3.3-70b-versatile", system_prompt="...")
    output = agent.run(user_message="Calculate the area of a circle with radius 2.")
    print(output)
"""

import json
import re
import logging

from groq import Groq

from src.agents.helpers.tool import Tool, validate_arguments
from src.agents.helpers.memory import ShortMemory
from src.utils.extraction import extract_tag_content
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ToolAgent:
    """
    The ToolAgent class represents an agent that can interact with a language model and use tools
    to assist with user queries. It generates function calls based on user input, validates arguments,
    and runs the respective tools.

    Attributes:
        tools (Tool | list[Tool]): A list of tools available to the agent.
        model (str): The model to be used for generating tool calls and responses.
        client (Groq): The Groq client used to interact with the language model.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool objects.
        system_prompt (str): The system prompt to be used for generating tool calls and responses.

    The system prompt construction logic ensures that:
        - If '{tools}' is present in the provided system_prompt, it is replaced with the actual tool signatures.
        - Otherwise, the prompt is used as-is.
        This allows for flexible prompt customization and robust tool injection for ReAct-style reasoning.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = None,
    ) -> None:
        self.client = Groq()
        self.model = model
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}
        if system_prompt is None:
            raise ValueError("ToolAgent requires a system_prompt to be provided by the consumer.")
        self.system_prompt = system_prompt  # assign before using
        if self.tools and "{tools}" in self.system_prompt:
            tool_signatures = "\n".join([tool.fn_signature for tool in self.tools])
            self.system_prompt = self.system_prompt.format(tools=tool_signatures)
        # Otherwise, assume the prompt is already formatted

    def run_node(self, tool_call_contents: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_call_contents (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_json in tool_call_contents:
            tool_call = json.loads(tool_call_json)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            logger.info(f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            logger.info(f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            logger.info(f"\nTool result: \n{result}")

            # Store the result using the tool call ID
            observations[validated_tool_call["id"]] = result

        return observations

    def run(
        self,
        user_message: str,
    ) -> str:
        """
        Handles the full process of interacting with the language model and executing a tool based on user input.

        Args:
            user_message (str): The user's message that prompts the tool agent to act.

        Returns:
            str: The final output after executing the tool and generating a response from the model.
        """
        user_prompt = {"role": "user", "content": user_message}

        tool_chat_history = ShortMemory(
            [
                {"role": "system", "content": self.system_prompt},
                user_prompt,
            ]
        )
        chat_history = ShortMemory([user_prompt])

        response = self.client.chat.completions.create(messages=tool_chat_history, model=self.model)
        model_response = str(response.choices[0].message.content)
        logger.debug(f"Raw model response:\n{model_response}")

        tool_calls = extract_tag_content(str(model_response), "tool_call")

        if tool_calls.found:
            observations = self.run_node(tool_calls.content)
            chat_history.append({"role": "user", "content": f'Observation: {observations}'})

        response = self.client.chat.completions.create(messages=chat_history, model=self.model)
        return str(response.choices[0].message.content)
