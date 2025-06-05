"""
self_reflection_agent.py

This module defines the SelfReflectionAgent class, which demonstrates how an agent can improve its answers by generating and critiquing its own responses in a loop. 
It is designed for beginners to learn about self-reflection and iterative improvement in LLM-powered agents.

Key Concepts:
- Self-reflection: The agent reviews and critiques its own output to make it better.
- Critique loop: The agent alternates between generating answers and reflecting on them until satisfied.
- Customizable prompts: You can provide your own prompts for generation and reflection.

Example usage:
    agent = SelfReflectionAgent(model="llama-3.3-70b-versatile")
    result = agent.run(human_input="Summarize the latest AI research.")
    print(result.final_candidate)
"""

import logging
from dotenv import load_dotenv
from groq import Groq
from collections import namedtuple

from src.agents.helpers.memory import ShortMemory
from src.utils.logging import get_logger

load_dotenv()

logger = get_logger(__name__)

class SelfReflectionAgent:
    """
    SelfReflectionAgent: Iteratively generates and critiques responses to optimize output quality.
    """

    DEFAULT_GENERATION_PROMPT = (
        "You are tasked with producing the best possible response to the user's request. "
        "If you receive feedback, revise your previous attempt accordingly. Always output the improved version."
    )
    DEFAULT_REFLECTION_PROMPT = (
        "Your job is to review the generated content and provide actionable feedback. "
        "If the content is flawless, respond with <SATISFIED>."
    )

    def __init__(self, model: str = "llama-3.3-70b-versatile", history_length: int = 3):
        self.client = Groq()
        self.model = model
        self.history_length = history_length

    def run(
        self,
        human_input: str,
        system_prompt: str = None,
        self_reflection_prompt: str = None,
        max_cycles: int = 10,
        verbose: int = 0,
        stop_token: str = "<SATISFIED>",
        stop_condition: callable = None,
        step_callback: callable = None,
    ):
        """
        Run the self-reflection loop.

        Args:
            human_input (str): The human's initial message or query.
            system_prompt (str, optional): Custom system prompt for generation.
            self_reflection_prompt (str, optional): Custom system prompt for self-reflection.
            max_cycles (int, optional): Maximum number of generate-reflect cycles.
            verbose (int, optional): Verbosity level for logging.
            stop_token (str, optional): Token indicating the critique loop should stop.
            stop_condition (callable, optional): Function that takes (candidate, feedback, trace) and returns True to stop.
            step_callback (callable, optional): Function called at each step with (cycle, candidate, feedback, trace).

        Returns:
            SelfReflectionResult: namedtuple with final_candidate and trace.
        """
        SelfReflectionResult = namedtuple("SelfReflectionResult", ["final_candidate", "observations"])

        generation_prompt = system_prompt or self.DEFAULT_GENERATION_PROMPT
        reflection_prompt = self_reflection_prompt or self.DEFAULT_REFLECTION_PROMPT

        generation_history = SelfReflectionChatHistory(
            [
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": human_input},
            ],
            total_length=self.history_length,
        )
        reflection_history = SelfReflectionChatHistory(
            [{"role": "system", "content": reflection_prompt}],
            total_length=self.history_length,
        )

        observations = []
        for cycle in range(max_cycles):
            if verbose:
                logger.info(f"\n{'=' * 50}")
                logger.info(f"STEP {cycle + 1}/{max_cycles}")
                logger.info(f"{'=' * 50}\n")

            response = self.client.chat.completions.create(messages=generation_history, model=self.model)
            generated_response = str(response.choices[0].message.content)
            if verbose:
                logger.info("\n" + "="*30 + " GENERATION " + "="*30)
                logger.info(f"{generated_response}")
                logger.info("="*72)
            generation_history.append({"role": "assistant", "content": generated_response})
            reflection_history.append({"role": "user", "content": generated_response})

            response = self.client.chat.completions.create(messages=reflection_history, model=self.model)
            feedback = str(response.choices[0].message.content)
            if verbose:
                logger.info("\n" + "-"*30 + " CRITIQUE " + "-"*30)
                logger.info(f"{feedback}")
                logger.info("-"*72)
            observations.append({"cycle": cycle, "generation": generated_response, "feedback": feedback})

            if step_callback:
                step_callback(cycle, generated_response, feedback, observations)

            should_stop = False
            if stop_condition:
                should_stop = stop_condition(generated_response, feedback, observations)
            else:
                should_stop = stop_token in feedback

            if should_stop:
                if verbose:
                    logger.info(f"\n\nStop token or condition met. Ending critique loop.\n\n")
                break

            generation_history.append({"role": "user", "content": feedback})
            reflection_history.append({"role": "assistant", "content": feedback})

        return SelfReflectionResult(final_candidate=generated_response, observations=observations)


class SelfReflectionChatHistory(ShortMemory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length, keeping the first message (usually the system prompt) always present.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """Add a message to the queue. The first message will always stay fixed (persistent prompt).

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)