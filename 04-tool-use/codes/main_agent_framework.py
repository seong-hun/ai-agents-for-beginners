import asyncio
import os
from typing import Annotated

import requests
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

load_dotenv()

SERP_API_KEY = os.getenv("SERP_API_KEY")
BASE_URL = "https://serpapi.com/search.json"


# Define Booking Plugin
class BookingPlugin:
    """Booking Plugin for customers"""

    def booking_hotel(
        self,
        query: Annotated[str, "The name of the city"],
        check_in_date: Annotated[str, "Hotel Check-in Time"],
        check_out_date: Annotated[str, "Hotel Check-out Time"],
    ) -> Annotated[str, "Return the result of booking hotel information"]:
        """Function to book a hotel.

        Parameters:
        - query: The name of the city
        - check_in_date: Hotel Check-in Time
        - check_out_date: Hotel Check-out Time

        Returns:
        - The result of booking hotel information
        """

        # Define the parameters for the hotel booking request
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": "1",
            "currency": "GBP",
            "gl": "uk",
            "hl": "en",
            "api_key": SERP_API_KEY,
        }

        # Send the GET request to the SERP API
        response = requests.get(BASE_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response content as JSON
            response = response.json()
            # Return the properties from the response
            return response["properties"]
        else:
            # Return None if the request failed
            return None

    def booking_flight(
        self,
        origin: Annotated[str, "The airport code of Departure"],
        destination: Annotated[str, "The airport code of Destination"],
        outbound_date: Annotated[str, "The date of outbound"],
        return_date: Annotated[str, "The date of Return_date"],
    ) -> Annotated[str, "Return the result of booking flight information"]:
        """Function to book a flight.

        Parameters:
        - origin: The airport code of Departure
        - destination: The airport_code of Destination
        - outbound_date: The date of outbound
        - return_date: The date of Return_date

        Returns:
        - The result of booking flight information
        """

        # Define the parameters for the outbound flight request
        go_params = {
            "engine": "google_flights",
            "departure_id": destination,
            "arrival_id": origin,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY,
        }

        print(go_params)

        # Send the GET request for the outbound flight
        go_response = requests.get(BASE_URL, params=go_params)

        # Initialize the result string
        result = ""

        # Check if the outbound flight request was successful
        if go_response.status_code == 200:
            # Parse the response content as JSON
            response = go_response.json()
            # Append the outbound flight information to the result
            result += "# outbound \n " + str(response)
        else:
            # Print an error message if the request failed
            print("error!!!")

        # Define the parameters for the return flight request
        back_params = {
            # "engine": "google_flights",
            "departure_id": destination,
            "arrival_id": origin,
            "outbound_date": outbound_date,
            "return_date": return_date,
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY,
        }

        # Send the GET request for the return flight
        back_response = requests.get(BASE_URL, params=back_params)

        # Check if the return flight request was successful
        if back_response.status_code == 200:
            # Parse the response content as JSON
            response = back_response.json()
            # Append the return flight information to the result
            result += "\n # return \n" + str(response)
        else:
            # Print an error message if the request failed
            print("error!!!")

        # Print the result
        print(result)

        # Return the result
        return result


async def main():
    async with (
        AzureCliCredential() as credential,
        AzureAIAgentClient(async_credential=credential) as client,
    ):
        # Define the agent's name and instructions
        AGENT_NAME = "BookingAgent"
        AGENT_INSTRUCTIONS = """
        You are a booking agent, help me to book flights or hotels.

        Thought: Understand the user's intention and confirm whether to use the reservation system to complete the task.

        Action:
        - If booking a flight, convert the departure name and destination name into airport codes.
        - If booking a hotel or flight, use the corresponding API to call. Ensure that the necessary parameters are available. If any parameters are missing, use default values or assumptions to proceed.
        - If it is not a hotel or flight booking, respond with the final answer only.
        - Output the results using a markdown table:
        - For flight bookings, separate the outbound and return contents and list them in the order of Departure_airport Name | Airline | Flight Number | Departure Time | Arrival_airport Name | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
        - For hotel bookings, list them in the order of Properties Name | Properties description | check_in_time | check_out_time | prices | nearby_places | hotel_class | gps_coordinates.
        """

        plugin = BookingPlugin()
        agent = client.create_agent(
            name=AGENT_NAME,
            instructions=AGENT_INSTRUCTIONS,
            tools=[plugin.booking_flight, plugin.booking_hotel],
        )

        user_inputs = [
            # "Can you tell me the round-trip air ticket from  London to New York JFK aiport, the departure time is February 17, 2025, and the return time is February 23, 2025"
            # "Book a hotel in New York from Feb 20,2025 to Feb 24,2025"
            "Help me book flight tickets and hotel for the following trip London Heathrow LHR Dec 20th 2025 to New York JFK returning Dec 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel"
            # "I have a business trip from London LHR to New York JFK on Feb 20th 2025 to Feb 27th 2025, can you help me to book a hotel and flight tickets"
        ]

        thread = agent.get_new_thread()

        # Process each user input
        for user_input in user_inputs:
            print(f"# User: '{user_input}'")
            # Get the agent's response for the specified thread
            response = await agent.run(messages=user_input, thread=thread)

            # Print the agent's response
            print(f"# {AGENT_NAME}: '{response.text}'")


if __name__ == "__main__":
    asyncio.run(main())
