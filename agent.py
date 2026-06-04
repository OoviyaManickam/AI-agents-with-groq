import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- STEP A: Define the actual Python function ---
def get_weather(city: str) -> str:
    # Fake data for now — real version would call a weather API
    weather_data = {
        "berlin": "18°C, cloudy",
        "london": "12°C, rainy",
        "tokyo": "28°C, sunny"
    }
    return weather_data.get(city.lower(), "Weather data not available for this city")

# --- STEP B: Describe the tool to the LLM ---
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


messages = [
    {"role": "user", "content": "What's the weather in Berlin?"}
]

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    tools=tools,
    max_tokens=512
)

print(response.choices[0].message)

