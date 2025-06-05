"""
memory.py

This module provides a simple memory class for storing chat or message history in agentic systems. 
It is designed for beginners to understand how agents can keep track of recent messages or context.

Key Concepts:
- ShortMemory: A list-like structure that holds a limited number of messages, automatically discarding the oldest when full.
- Useful for chatbots, agents, or any system that needs to remember recent interactions.

Example usage:
    memory = ShortMemory(total_length=3)
    memory.append('Hello')
    memory.append('How are you?')
    memory.append('What is your name?')
    memory.append('Tell me a joke.')
    print(memory)  # Output: ['How are you?', 'What is your name?', 'Tell me a joke.']
"""

class ShortMemory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)
