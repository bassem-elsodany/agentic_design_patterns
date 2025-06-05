"""
react_agent.py

This module defines the ReactAgent class, which implements the ReAct (Reasoning + Acting) pattern for LLM-powered agents. 
It is designed for beginners to learn how agents can reason step-by-step, use tools, and interact with users in a loop.

Key Concepts:
- ReAct pattern: Alternates between reasoning (thoughts) and actions (tool calls) to solve tasks.
- Tool use: Agents can call external functions to get information or perform operations.
- Multi-step reasoning: Agents can chain thoughts and actions until a final answer is produced.

Example usage:
    agent = ReactAgent(tools=[...], model="llama-3.3-70b-versatile", system_prompt="...")
    output = agent.run(user_message="What's the weather in Paris?")
    print(output)
"""

import json
import re
import logging
from dotenv import load_dotenv
from groq import Groq

from src.agents.helpers.tool import Tool, validate_arguments
from src.agents.helpers.memory import ShortMemory
from src.utils.extraction import extract_tag_content
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

class ReactAgent:
    """
    A class that represents an agent using the ReAct logic that interacts with tools to process
    user inputs, make decisions, and execute tool calls. The agent can run interactive sessions,
    collect tool signatures, and process multiple tool calls in a given round of interaction.

    Attributes:
        client (Groq): The Groq client used to handle model-based completions.
        model (str): The name of the model used for generating responses. Default is "llama-3.3-70b-versatile".
        tools (list[Tool]): A list of Tool instances available for execution.
        tools_dict (dict): A dictionary mapping tool names to their corresponding Tool instances.

    Note:
        The system prompt provided to the agent should include the placeholder '{tools}'
        within <tools>...</tools> tags. This placeholder will be automatically replaced
        with the actual tool signatures at runtime, ensuring the agent is aware of the available tools.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "llama-3.3-70b-versatile",
        system_prompt: str = None,
    ) -> None:
        """
        Initialize the ReactAgent.

        Args:
            tools (Tool | list[Tool]): A single Tool or a list of Tool instances to be made available to the agent.
            model (str): The model name to use for completions.
            system_prompt (str): The system prompt template. It should include '{tools}' as a placeholder
                within <tools>...</tools> tags, which will be replaced with the actual tool signatures.

        The system prompt construction logic ensures that:
        - If '{tools}' is present in the provided system_prompt, it is replaced with the actual tool signatures.
        - Otherwise, the prompt is used as-is.
        This allows for flexible prompt customization and robust tool injection for ReAct-style reasoning.
        """
        self.client = Groq()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

        if self.tools and "{tools}" in self.system_prompt:
            tool_signatures = "\n".join([tool.fn_signature for tool in self.tools])
            self.system_prompt = self.system_prompt.format(tools=tool_signatures)
        # Otherwise, assume the prompt is already formatted and do nothing

    def run(
        self,
        user_message: str,
        max_rounds: int = 10,
    ) -> str:
        """
        Executes a user interaction session using the ReAct protocol.
        """
        user_prompt = {"role": "user", "content": f"<question>{user_message}</question>"}
        chat_history = ShortMemory(
            [
                {"role": "system", "content": self.system_prompt},
                user_prompt,
            ]
        )

        if self.tools:
            for _ in range(max_rounds):
                logger.info(f"\n{'=' * 50}")
                logger.info(f"ROUND {_ + 1}/{max_rounds}")
                logger.info(f"{'=' * 50}\n")

                response = self.client.chat.completions.create(
                    messages=chat_history, model=self.model
                )
                model_response = str(response.choices[0].message.content)
                logger.debug(f"Raw model response:\n{model_response}")

                response = extract_tag_content(model_response, "response")
                if response.found:
                    return response.content[0]

                # Always append the model's output
                chat_history.append({"role": "assistant", "content": model_response})

                thought = extract_tag_content(model_response, "thought")
                if thought.found:
                    logger.info(f"\nThought: {thought.content[0]}")

                tool_calls = extract_tag_content(model_response, "tool_call")
                if tool_calls.found:
                    # Execute actual tools
                    observations = self.run_node(tool_calls.content)
                    logger.info(f"\nObservations: {observations}")
                    chat_history.append({
                        "role": "user",
                        "content": f"<observation>{json.dumps(observations)}</observation>"
                    })
                    continue  # Let the model respond to the observation

        response = self.client.chat.completions.create(messages=chat_history, model=self.model)            
        return str(response.choices[0].message.content)

    def run_node(self, tool_calls_content: list) -> dict:
        """
        Processes each tool call, validates arguments, executes the tools, and collects results.

        Args:
            tool_calls_content (list): List of strings, each representing a tool call in JSON format.

        Returns:
            dict: A dictionary where the keys are tool call IDs and values are the results from the tools.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
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
