"""
demo_tools.py

This module provides example tools (functions) for use with agentic agents. 
It is designed for beginners to experiment with tool use, LLM integration, and agent reasoning in a safe, easy-to-understand way.

Key Concepts:
- Tool: A function that an agent can call to get information or perform a calculation.
- Decorator: The @tool decorator makes a function available to agents.
- Logging: Each tool logs its usage for transparency and debugging.

Example usage:
    from demo_tools import get_weather
    print(get_weather.run(city="Paris"))  # Output: Sunny, 22°C
"""

from src.agents.helpers.tool import tool
from utils.logging import get_logger

logger = get_logger("DemoTools")

@tool
def get_weather(city: str) -> str:
    """
    get_weather(city: str) -> str
    Returns the current weather for a given city.
    
    Args:
        city (str): The name of the city to get the weather for.
    
    Returns:
        str: A string describing the weather in the specified city.
    """
    logger.debug(f"get_weather called with city={city}")
    weather_data = {
        "Paris": "Sunny, 22°C",
        "London": "Cloudy, 18°C",
        "New York": "Rainy, 16°C"
    }
    return weather_data.get(city, "Weather data not available.")

@tool
def calculate_area(radius: float) -> float:
    """
    calculate_area(radius: float) -> float
    Calculates the area of a circle given its radius.
    
    Args:
        radius (float): The radius of the circle in meters.
    
    Returns:
        float: The area of the circle in square meters.
    """
    logger.debug(f"calculate_area called with radius={radius}")
    import math
    return math.pi * radius * radius

@tool
def recommend_food(city: str) -> str:
    """
    recommend_food(city: str) -> str
    Returns a list of recommended picnic foods for a given city.
    
    Args:
        city (str): The name of the city for which to recommend foods.
    
    Returns:
        str: A comma-separated list of recommended picnic foods for the city.
    """
    logger.debug(f"recommend_food called with city={city}")
    food_data = {
        "Paris": "Baguette, cheese, grapes, macarons",
        "London": "Sandwiches, scones, strawberries, tea",
        "New York": "Bagels, pretzels, cheesecake, hot dogs"
    }
    return food_data.get(city, "No food recommendations available.") 

@tool
def get_planet_distance(planet: str) -> str:
    """
    Returns the average distance from Earth to a specified planet.

    Args:
        planet (str): The name of the planet to look up (e.g., "Mars", "Venus").

    Returns:
        str: The average distance from Earth to the given planet as a string (e.g., "225 million km").
             If the planet is not recognized, returns "Unknown".

    Example:
        >>> get_planet_distance("Mars")
        "225 million km"

    Supported planets:
        - Mars
        - Venus

    Usage:
        Use this tool to find out how far a planet is from Earth on average.
    """
    distances = {"Mars": "225 million km", "Venus": "261 million km"}
    logger.debug(f"get_planet_distance called with planet={planet}")
    return distances.get(planet, "Unknown")