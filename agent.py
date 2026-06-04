import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Tool function
def get_weather(city: str) -> str:
    weather_data = {
        "berlin": "18°C, cloudy",
        "london": "12°C, rainy",
        "tokyo": "28°C, sunny"
    }
    return weather_data.get(city.lower(), "Weather data not available for this city")

# Map tool names to actual functions
# This lets us call any tool by name dynamically
available_tools = {
    "get_weather": get_weather
}

# Tool descriptions for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Start conversation
messages = [
    {"role": "user", "content": "What's the weather in Berlin and tokyo?"}
]

print("Starting agent loop...\n")

# THE AGENT LOOP
while True:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        max_tokens=512
    )

    response_message = response.choices[0].message

    # Case 1: LLM wants to call a tool
    if response_message.tool_calls:
        # Add LLM's response to message history
        messages.append(response_message)

        # Process each tool call (LLM can request multiple at once)
        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_inputs = json.loads(tool_call.function.arguments)

            print(f"Tool called: {tool_name}")
            print(f"Inputs: {tool_inputs}")

            # Run the actual Python function
            tool_result = available_tools[tool_name](**tool_inputs)

            print(f"Result: {tool_result}\n")

            # Feed result back into message history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

    # Case 2: LLM gave a final text answer — we're done
    else:
        print("Final answer:")
        print(response_message.content)
        break