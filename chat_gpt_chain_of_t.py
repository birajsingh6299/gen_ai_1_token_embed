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

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is 2 + 2 * 2 / 2 * 5 ?"},

        {"role": "assistant", "content": json.dumps({"step": "analyse",
                                                     "content": "The user is interested in a maths query involving a sequence of operations including addition, multiplication, and division."})},
        {"role": "assistant", "content": json.dumps(
            {"step": "think", "content": "The expression 2 + 2 * 2 / 2 * 5 must be solved following the order of operations (PEMDAS/BODMAS), which stands for Parentheses, Exponents, Multiplication and Division (from left to right), and Addition and Subtraction (from left to right)."})},
        {"role": "assistant", "content": json.dumps(
            {"step": "output", "content": "12"})},
        {"role": "assistant", "content": json.dumps(
            {"step": "validate", "content": "By following the order of operations: 2 + ((2 * 2) / 2) * 5 = 2 + (4 / 2) * 5 = 2 + 2 * 5 = 2 + 10 = 12. Hence, the calculation is correct."})}
    ]
)

print(response.choices[0].message.content)
