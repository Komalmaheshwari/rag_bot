from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from dotenv import load_dotenv

print("Loading environment variables...")

# Load environment variables from .env file (optional)
load_dotenv()

# Optional LangChain API key (for tracing, analytics)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question as best you can."),
    ("human", "{question}")
])

# Streamlit UI
st.title('Fun Chatbot for Practice')
user_input = st.text_input("Search the topic you want:")

# Gemini model from Google Generative AI
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key="AIzaSyAAab-neYnQvKPqh46OsWrjzw0itfCoowQ"
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Run the chain
if user_input:
    response = chain.invoke({"question": user_input})
    st.write(response)
