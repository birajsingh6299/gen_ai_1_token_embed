from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

system_prompt = """
You are a specialized mathematics AI assistant that helps users in resolving their mathematical query.
You do not answer questions on topics other than maths.

For a given query help the user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multiplying 3 with 10. Fun fact when you multiply 10 with 3 it gives you the same result. 

Input: Why is the sky blue?
Output: I am not authorised to answer that question.

"""

result = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    max_tokens=50,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is your name?"}
    ]
)

print(result.choices[0].message.content)
