import asyncio
import os
from random import randint
from typing import Awaitable, Callable

from agent_framework import AgentThread, FunctionInvocationContext
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()


def get_random_destination() -> str:
    """Get a random vacation destination using Repository Pattern.

    This function exemplifies several design patterns:
    - Strategy Pattern: Interchangeable algorithm for destination selection
    - Repository Pattern: Encapsulates data access logic
    - Factory Method: Creates destination objects on demand

    Returns:
        str: A randomly selected destination following consistent format
    """
    # Data Repository Pattern: Centralized destination data management
    destinations = [
        "Barcelona, Spain",  # Mediterranean cultural hub
        "Paris, France",  # European artistic center
        "Berlin, Germany",  # Historical European capital
        "Tokyo, Japan",  # Asian technology metropolis
        "Sydney, Australia",  # Oceanic coastal city
        "New York, USA",  # American urban center
        "Cairo, Egypt",  # African historical capital
        "Cape Town, South Africa",  # African scenic destination
        "Rio de Janeiro, Brazil",  # South American beach city
        "Bali, Indonesia",  # Southeast Asian island paradise
    ]

    # Factory Method Pattern: Create destination selection on demand
    return destinations[randint(0, len(destinations) - 1)]


async def handle_streaming_intermediate_steps(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    print(
        f"Function Call:> {context.function.name} with arguments: {context.arguments}"
    )
    print()
    await next(context)
    print(f"Function Result:> {context.result}")
    print()


openai_chat_client = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id=os.environ.get("GITHUB_MODEL_ID"),
)

AGENT_NAME = "TravelAgent"

AGENT_INSTRUCTIONS = """You are a helpful AI Agent that can help plan vacations for customers.

Important: When users specify a destination, always plan for that location. Only suggest random destinations when the user hasn't specified a preference.

When the conversation begins, introduce yourself with this message:
"Hello! I'm your TravelAgent assistant. I can help plan vacations and suggest interesting destinations for you. Here are some things you can ask me:
1. Plan a day trip to a specific location
2. Suggest a random vacation destination
3. Find destinations with specific features (beaches, mountains, historical sites, etc.)
4. Plan an alternative trip if you don't like my first suggestion

What kind of trip would you like me to help you plan today?"

Always prioritize user preferences. If they mention a specific destination like "Bali" or "Paris," focus your planning on that location rather than suggesting alternatives.
"""

agent = OpenAIChatClient(
    base_url=os.environ.get("GITHUB_ENDPOINT"),
    api_key=os.environ.get("GITHUB_TOKEN"),
    model_id=os.environ.get("GITHUB_MODEL_ID"),
).create_agent(
    name=AGENT_NAME,
    instructions=AGENT_INSTRUCTIONS,
    tools=[get_random_destination],
    middleware=[handle_streaming_intermediate_steps],
)


async def process_input(
    user_input: str | None,
    thread: AgentThread | None,
) -> AgentThread | None:
    first_chunk = True
    async for update in agent.run_stream(user_input, thread=thread):
        if update.text:
            if first_chunk:
                print(f"> {'Assistant'}: ", end="", flush=True)
                first_chunk = False
            print(update.text, end="", flush=True)
    print("\n")


async def main():
    thread = agent.get_new_thread()

    await process_input("Hello", thread)

    while True:
        user_input = input("> User: ")
        if user_input == "exit":
            break
        print()

        await process_input(user_input, thread)


if __name__ == "__main__":
    asyncio.run(main())
