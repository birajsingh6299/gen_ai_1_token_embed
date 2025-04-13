from openai import OpenAI
from dotenv import load_dotenv
import json
import requests
import os

load_dotenv()

client = OpenAI()


def run_command(command):
    result = os.system(command=command)
    return result


def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "Something went wrong"


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for that city."
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as an input and runs it for the user."
    }
}
system_prompt = """
    You are a helpful AI assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tool and plan the step by step execution, based on the planning,
    select the relevant tool from the available tool and based on the tool selection you perform an action of calling the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON format:
    {{
        "step":"string",
        "content":"string",
        "function":"The name of function if the step is action",
        "input":"The input parameter for the function"
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for that city.
    - run_command: Takes a command as an input and runs it for the user.
    
    Example:
    User Query: What is the weather of New York?
    Output: {{ "step":"plan",
        "content":"The user is interested in getting the current weather of New York" }}
    Output: {{ "step":"plan",
        "content":"From the available tools I should call get_weather" }}
    Output: {{ "step":"action", "function":"get_weather", "input":"New York" }}
    Output: {{ "step":"observe", "output":"12 Degree Celcius" }}
    Output: {{ "step":"output",
        "content":"The weather for New York seems to be 12 Degree Celcius" }}

"""

messages = [{"role": "system", "content": system_prompt}]

while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    while True:

        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append(
            {"role": "assistant", "content": json.dumps(parsed_output)})

        if parsed_output.get("step") == "plan":
            print(f"🤯: {parsed_output.get("content")}")
            continue

        if parsed_output.get("step") == "action":
            print(f"🤯: {parsed_output}")
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name):
                output = available_tools.get(tool_name).get("fn")(tool_input)
                messages.append({"role": "assistant", "content": json.dumps(
                    {"step": "observe", "output": output})})
                continue

        if parsed_output.get("step") == "output":
            print(f"🚆:{parsed_output.get("content")}")
            break
