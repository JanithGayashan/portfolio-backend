import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Downloading free BAAI tokenization model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

resume_chunks = [
    "Janith Gayashan is an AI Engineer and student pursuing a Bachelor of Science (Hons) in Artificial Intelligence at the University of Moratuwa (Batch 21).",
    "Janith completed a 5-month internship at SLT Digital Lab where he contributed to a real-world AI project building the slt-travel-chatbot.",
    "Janith's technical stack includes Python, FastAPI, React, Next.js, LangGraph, Pinecone, OpenAI, and Scikit-Learn.",
    "Janith built a production-ready Loan Approval Decision Tree model and deployed it via a FastAPI microservice."
]

documents = [Document(page_content=text) for text in resume_chunks]

print("Uploading vectors to Pinecone...")
PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name="portfolio-index"
)
print("✅ Database successfully seeded!")