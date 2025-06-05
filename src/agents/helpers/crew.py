"""
crew.py

This module provides the Crew class, which helps organize and manage a group of agents working together in a multi-agent system. 
It is designed for beginners interested in agentic design patterns, showing how agents can be registered, ordered by dependencies, and run as a team.

Key Concepts:
- Crew: A container for agents that manages their execution order and dependencies.
- Topological sort: Ensures agents are run in the correct order based on their dependencies.
- Visualization: Supports plotting agent relationships as a graph.

Example usage:
    with Crew() as crew:
        crew.add_agent(agent1)
        crew.add_agent(agent2)
        crew.run()
"""

from collections import deque

from graphviz import Digraph  # type: ignore

from utils.logging import get_logger

logger = get_logger("Crew")


class Crew:
    """
    Manages a collection of agents and their execution order in collaborative workflows.

    The Crew class organizes agents, tracks their dependencies, and provides utilities for running them in dependency-respecting order. It also supports context management for agent registration and visualization of agent relationships.

    Attributes:
        current_crew (Crew): The currently active Crew context (for context manager support).
        agents (list): The list of agents managed by this crew.
    """

    current_crew = None

    def __init__(self):
        self.agents = []

    def __enter__(self):
        """
        Enters the context manager, setting this crew as the current active context.

        Returns:
            Crew: The current Crew instance.
        """
        Crew.current_crew = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exits the context manager, clearing the active context.

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The traceback, if an exception was raised.
        """
        Crew.current_crew = None

    def add_agent(self, agent):
        """
        Adds an agent to the crew.

        Args:
            agent: The agent to be added to the crew.
        """
        self.agents.append(agent)

    @staticmethod
    def register_agent(agent):
        """
        Registers an agent with the current active crew context.

        Args:
            agent: The agent to be registered.
        """
        if Crew.current_crew is not None:
            Crew.current_crew.add_agent(agent)

    def topological_sort(self):
        """
        Performs a topological sort of the agents based on their dependencies.

        Returns:
            list: A list of agents sorted in topological order.

        Raises:
            ValueError: If there's a circular dependency among the agents.
        """
        in_degree = {agent: len(agent.dependencies) for agent in self.agents}
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])

        sorted_agents = []

        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)

            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_agents) != len(self.agents):
            raise ValueError(
                "Circular dependencies detected among agents, preventing a valid topological sort"
            )

        return sorted_agents

    def plot(self):
        """
        Plots the Directed Acyclic Graph (DAG) of agents in the crew using Graphviz.

        Returns:
            Digraph: A Graphviz Digraph object representing the agent dependencies.
        """
        dot = Digraph(format="png")  # Set format to PNG for inline display

        # Add nodes and edges for each agent in the crew
        for agent in self.agents:
            dot.node(agent.name)
            for dependency in agent.dependencies:
                dot.edge(dependency.name, agent.name)
        return dot

    def run(self):
        """
        Runs all agents in the crew in topologically sorted order.

        This method executes each agent's run method and logs the results.
        """
        sorted_agents = self.topological_sort()
        for agent in sorted_agents:
            logger.info(f"RUNNING AGENT: {agent}")
            result = agent.run()
            logger.info(f"{result}")
