"""
tool.py

This module provides utilities for defining and using tools (functions) in agentic systems. 
A tool is a function that an agent can call to perform a specific task, such as looking up information or performing a calculation.

Key Concepts:
- Tool: A wrapper around a function, making it easy for agents to discover and use it.
- Signature extraction: Automatically generates a description and argument types for each tool.
- Validation: Ensures arguments passed to tools are of the correct type.

Example usage:
    @tool
    def add(a: int, b: int) -> int:
        'Add two numbers.'
        return a + b
    result = add.run(a=2, b=3)
    print(result)  # Output: 5
"""

import json
from typing import Callable
from functools import wraps


def get_fn_signature(fn: Callable) -> dict:
    """
    Generates the signature for a given function.

    Args:
        fn (Callable): The function whose signature needs to be extracted.

    Returns:
        dict: A dictionary containing the function's name, description,
              and parameter types.
    """
    # Use the original function if available (for decorated functions)
    original = getattr(fn, 'original_fn', fn)
    fn_signature: dict = {
        "name": original.__name__,
        "description": original.__doc__,
        "parameters": {"properties": {}},
    }
    schema = {
        k: {"type": v.__name__} for k, v in original.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary to match the expected types.

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool.
        tool_signature (dict): The expected function signature and parameter types.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
    """
    properties = tool_signature["parameters"]["properties"]

    # TODO: This is overly simplified but enough for simple Tools.
    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }

    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")

        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call


class Tool:
    """
    A class representing a tool that wraps a callable and its signature.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents.
        fn_signature (str): JSON string representation of the function's signature.
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self.fn_signature = json.dumps(get_fn_signature(fn))
        self.original_fn = fn
        wraps(fn)(self)

    def run(self, *args, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        return self.fn(*args, **kwargs)


def tool(fn: Callable) -> Tool:
    """
    Decorates a function to return a Tool object with a .run() method.

    Args:
        fn (Callable): The function to be decorated.

    Returns:
        Tool: A Tool object wrapping the function.
    """
    return Tool(fn)
