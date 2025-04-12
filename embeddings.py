from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

text = "Eiffel tower is in Paris and is a famous landmark, it is 324 meters tall"

response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

print(response)
# print("Vector Embeddings", response.data[0].embedding)
