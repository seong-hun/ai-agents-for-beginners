import asyncio
import os
import random
from typing import Annotated

import dotenv
from openai import AsyncOpenAI
from semantic_kernel.agents import (
    AgentThread,
    ChatCompletionAgent,
)
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import (
    ChatMessageContent,
    FunctionCallContent,
    FunctionResultContent,
)
from semantic_kernel.functions import kernel_function

dotenv.load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/",
)

chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

# --- Plugin


class DestinationPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        # List of vacation destinations
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia",
        ]
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(
        self,
    ) -> Annotated[str, "Returns a random vacation destination."]:
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        destination = random.choice(available_destinations)

        self.last_destination = destination

        return destination


# --- Create an agent

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

agent = ChatCompletionAgent(
    service=chat_completion_service,
    name="TravelAgent",
    instructions=AGENT_INSTRUCTIONS,
    plugins=[DestinationPlugin()],
)


# --- Conversation loop


async def handle_streaming_intermediate_steps(message: ChatMessageContent) -> None:
    for item in message.items or []:
        if isinstance(item, FunctionResultContent):
            print(f"\nFunction Result:> {item.result} for function: {item.name}")
            print()
        elif isinstance(item, FunctionCallContent):
            print()
            print(f"Function Call:> {item.name} with arguments: {item.arguments}")
        else:
            print(f"{item}")


async def process_input(
    user_input: str | None,
    thread: AgentThread | None,
) -> AgentThread | None:
    first_chunk = True
    async for response in agent.invoke_stream(
        messages=user_input,
        thread=thread,
        on_intermediate_message=handle_streaming_intermediate_steps,
    ):
        thread = response.thread
        if first_chunk:
            print(f"> {response.name or 'Assistant'}: ", end="", flush=True)
            first_chunk = False
        print(response.content, end="", flush=True)
    print("\n")

    return thread


async def main():
    thread: AgentThread | None = None

    try:
        thread = await process_input("Hello", thread)

        while True:
            user_input = input("> User: ")
            if user_input == "exit":
                break

            thread = await process_input(user_input, thread)
    finally:
        await thread.delete() if thread else None


if __name__ == "__main__":
    asyncio.run(main())
