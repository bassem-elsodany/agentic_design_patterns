"""
multi_agent.py

This module defines the Agent class for building collaborative, multi-agent systems. 
It is designed for beginners to learn how agents can work together, share context, and solve complex tasks as a team.

Key Concepts:
- Agent: An autonomous entity that can perform tasks, use tools, and communicate with other agents.
- Dependencies: Agents can depend on each other, forming a workflow or pipeline.
- Context sharing: Agents can pass information to each other to improve results.

Example usage:
    agent = Agent(
        name="Researcher",
        backstory="An expert in gathering information.",
        task_description="Find the latest research on climate change.",
        tools=[...]
    )
    output = agent.run()
    print(output)
"""

import logging
from textwrap import dedent

from src.agents.helpers.crew import Crew
from src.agents.react_agent import ReactAgent
from src.agents.helpers.tool import Tool
from src.utils.logging import get_logger

logger = get_logger(__name__)

class Agent:
    """
    An autonomous agent designed for collaborative, multi-agent workflows.

    This class models an AI agent that can participate in a team, manage dependencies, share context, and execute assigned tasks. Agents can be chained or grouped to solve complex problems by passing information and results between each other.

    Attributes:
        name (str): Identifier for the agent.
        backstory (str): Background or persona for the agent.
        task_description (str): The main objective or assignment for the agent.
        task_expected_output (str): The required format or content for the agent's output.
        react_agent (ReactAgent): The underlying LLM-powered agent used for reasoning and tool use.
        dependencies (list[Agent]): Other agents this agent relies on.
        dependents (list[Agent]): Agents that depend on this agent's output.
        context (str): Shared or received information from other agents.

    Args:
        name (str): The agent's name.
        backstory (str): Persona or background for the agent.
        task_description (str): The agent's assigned task.
        task_expected_output (str, optional): Output requirements. Defaults to "".
        tools (list[Tool] | None, optional): Tools available to the agent. Defaults to None.
        llm (str, optional): Language model to use. Defaults to "llama-3.3-70b-versatile".
    """

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        tools: list[Tool] | None = None,
        llm: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize a new Agent for use in a multi-agent system.

        This constructor sets up the agent's identity, its assigned task, any required output format, and the tools and language model it will use. It also prepares the agent for collaboration by initializing dependency and context tracking.

        Args:
            name (str): Unique name for the agent.
            backstory (str): Persona or background information for the agent.
            task_description (str): The main task or goal assigned to the agent.
            task_expected_output (str, optional): The required output format or content. Defaults to an empty string.
            tools (list[Tool] | None, optional): Tools the agent can use. Defaults to None.
            llm (str, optional): The language model to use for reasoning. Defaults to "llama-3.3-70b-versatile".
        """
        self.name = name
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=tools or []
        )

        self.dependencies: list[Agent] = []  # Agents that this agent depends on
        self.dependents: list[Agent] = []  # Agents that depend on this agent

        self.context = ""

        # Automatically register this agent to the active Crew context if one exists
        Crew.register_agent(self)

    def __repr__(self):
        return f"{self.name}"

    def __rshift__(self, other):
        """
        Defines the '>>' operator. This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that depends on this agent.
        """
        self.add_dependent(other)
        return other  # Allow chaining

    def __lshift__(self, other):
        """
        Defines the '<<' operator to indicate agent dependency in reverse.

        Args:
            other (Agent): The agent that this agent depends on.

        Returns:
            Agent: The `other` agent to allow for chaining.
        """
        self.add_dependency(other)
        return other  # Allow chaining

    def __rrshift__(self, other):
        """
        Defines the '<<' operator.This operator is used to indicate agent dependency.

        Args:
            other (Agent): The agent that this agent depends on.
        """
        self.add_dependency(other)
        return self  # Allow chaining

    def __rlshift__(self, other):
        """
        Defines the '<<' operator when evaluated from right to left.
        This operator is used to indicate agent dependency in the normal order.

        Args:
            other (Agent): The agent that depends on this agent.

        Returns:
            Agent: The current agent (self) to allow for chaining.
        """
        self.add_dependent(other)
        return self  # Allow chaining

    def add_dependency(self, other):
        """
        Adds a dependency to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that this agent depends on.

        Raises:
            TypeError: If the dependency is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            self.dependencies.append(other)
            other.dependents.append(self)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                self.dependencies.append(item)
                item.dependents.append(self)
        else:
            raise TypeError("The dependency must be an instance or list of Agent.")

    def add_dependent(self, other):
        """
        Adds a dependent to this agent.

        Args:
            other (Agent | list[Agent]): The agent(s) that depend on this agent.

        Raises:
            TypeError: If the dependent is not an Agent or a list of Agents.
        """
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list) and all(isinstance(item, Agent) for item in other):
            for item in other:
                item.dependencies.append(self)
                self.dependents.append(item)
        else:
            raise TypeError("The dependent must be an instance or list of Agent.")

    def receive_context(self, input_data):
        """
        Receives and stores context information from other agents.

        Args:
            input_data (str): The context information to be added.
        """
        self.context += f"{self.name} received context: \n{input_data}"

    def create_prompt(self):
        """
        Creates a prompt for the agent based on its task description, expected output, and context.

        Returns:
            str: The formatted prompt string.
        """
        prompt = dedent(
            f"""
        You are an AI agent. You are part of a team of agents working together to complete a task.
        I'm going to give you the task description enclosed in <task_description></task_description> tags. I'll also give
        you the available context from the other agents in <context></context> tags. If the context
        is not available, the <context></context> tags will be empty. You'll also receive the task
        expected output enclosed in <task_expected_output></task_expected_output> tags. With all this information
        you need to create the best possible response, always respecting the format as describe in
        <task_expected_output></task_expected_output> tags. If expected output is not available, just create
        a meaningful response to complete the task.

        <task_description>
        {self.task_description}
        </task_description>

        <task_expected_output>
        {self.task_expected_output}
        </task_expected_output>

        <context>
        {self.context}
        </context>

        Your response:
        """
        ).strip()

        return prompt

    def run(self):
        """
        Runs the agent's task and generates the output.

        This method creates a prompt, runs it through the ReactAgent, and passes the output to all dependent agents.

        Returns:
            str: The output generated by the agent.
        """
        msg = self.create_prompt()
        logger.info(f"\n[MultiAgent] Prompt:\n{msg}")
        output = self.react_agent.run(user_message=msg)
        logger.info(f"\n[MultiAgent] Output:\n{output}")

        # Pass the output to all dependents
        for dependent in self.dependents:
            logger.info(f"\n[MultiAgent] Passing context to {dependent.name}")
            dependent.receive_context(output)
        return output
