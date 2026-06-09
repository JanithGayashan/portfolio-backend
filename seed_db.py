import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

print("Downloading free BAAI tokenization model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 1. Load the private text file
print("Reading private resume data...")
data_path = os.path.join(os.path.dirname(__file__), "data", "resume.txt")

try:
    loader = TextLoader(data_path)
    documents = loader.load()
except FileNotFoundError:
    print(f"❌ Error: Could not find the file at {data_path}")
    print("Please ensure you created the 'data/resume.txt' file!")
    exit(1)

# 2. Split the document into chunks (It splits by the double blank lines we left)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(documents)

# 3. Upload to Pinecone
print(f"Uploading {len(docs)} vectors to Pinecone...")
PineconeVectorStore.from_documents(
    docs, 
    embeddings, 
    index_name="portfolio-index"
)
print("✅ Database successfully seeded!")