# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Ensure API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not set in environment or .env")

# Initialize FastAPI app
app = FastAPI(
    title="Chroma Chat with OpenAI",
    description="Chat with retrieved context using OpenAI ChatCompletion",
    version="1.1.0"
)

# Enable CORS (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector store and embedding model
embedding = OpenAIEmbeddings()
db = Chroma(persist_directory="chroma_db", embedding_function=embedding)

# Initialize OpenAI Chat Model
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

# Request schema
class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

# Response schema
class ChatResponse(BaseModel):
    answer: str
    context_used: list[str]
    updated_history: list[dict]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query
    history = request.history

    # Step 1: Retrieve relevant context
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 2: Build the messages for OpenAI
    messages = []

    # Add system prompt to instruct LLM to use context
    messages.append({
        "role": "system",
        "content": f"You are a helpful assistant. Use the following context to answer the user's question:\n\n{context}"
    })

    # Add chat history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add user query
    messages.append({"role": "user", "content": query})

    # Step 3: Generate response from OpenAI
    response = chat_model.invoke(messages)
    answer = response.content

    # Step 4: Update history
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]

    return {
        "answer": answer,
        "context_used": [doc.page_content for doc in docs],
        "updated_history": updated_history
    }
