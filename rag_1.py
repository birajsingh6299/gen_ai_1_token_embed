from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os
import json


load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# vector_store.add_documents(documents=split_docs)
print("Injection Done")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

client = OpenAI()

system_prompt = """
You are a helpful AI assistant, 
that understands the user query and answers it by referring to the information provided.
For every user query there will be a chunk of information provided, you need to answer the user query
based on the information provided.

Rules:
    - Follow the Output Text Format.
    - Always answer from the input PDF content only.
    - Carefully analyse the user query
    - If the relevant information is not found in the reference pdf chunk, then return no relevant information found.

Example:

    User Query: What is FS module?

    Input: {{"role":"assistant", "content":"<information to refer to>"}}
    Input: {{"role":"assistant", "content":"What is FS module?"}}
    
    Output: The built-in Node.js file system module helps us store, access, and manage data on our operating system. 
    Commonly used features of the fs module include fs.readFile to read data from a file, fs.writeFile to write data to a file and replace the file if it already exists, fs.watchFile to get notified of changes, and fs.appendFile to append data to a file. 
    The fs core module is available in every Node.js project without having to install it.

"""
messages = [{"role": "system", "content": system_prompt}]
while True:
    user_query = input("> ")

    search_result = retriever.similarity_search(
        query=user_query,
    )

    page_contents = list(map(lambda x: x.page_content, search_result))

    page_contents_str = ''.join(page_contents)

    messages.extend([{"role": "assistant", "content": page_contents_str}, {
                    "role": "user", "content": user_query}])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content})
    print(response.choices[0].message.content)
