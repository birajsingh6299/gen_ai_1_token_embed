from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant that solves the user's complex problems by breaking it down into several steps.

1. Taking the user input.
2. Analysing what is being asked by the user.
3. Thinking on how the query can be resolved?
4. Getting the output.
5. Validating the output.
6. Returning the result

Rules:
1. Follow the strict JSON output as per the output schema.
2. Always perform one step at a time and wait for next input.
3. Carefully analyse the user query.

Output Format:
{{step:"string", content:"string"}}

Example:
Input: What is 2 + 2 ?
Output: {{ step: "analyse", content: "Alright! The user is interested in a maths query. He is asking for a basic arithmatic operation."}}
Output: {{ step: "think", content: "The user wants to add 2 with 2 and get the result. Since it is an addition operation it will be performed from left to right."}}
Output: {{ step: "output", content: "4"}}
Output: {{ step: "validate", content: "Seems like 4 is the correct answer for 2 + 2"}}
Output: {{ step: "result", content: "The result of 2 + 2 is 4. Is there anything else that you would like me to help with?"}}

"""

messages = [
    {"role": "system", "content": system_prompt},
]

user_input = input("> ")

messages.append({"role": "user", "content": user_input})

while True:

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )

    parse_response = json.loads(response.choices[0].message.content)

    if parse_response.get("step") != "result":
        print(f"🧠:{parse_response}")
        messages.append(
            {"role": "assistant", "content": json.dumps(parse_response)})
        continue

    print(f"✅:{parse_response}")
    break
