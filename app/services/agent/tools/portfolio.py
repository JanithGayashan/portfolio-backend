import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.api.v1.loan import predict_loan_status
from app.schemas.loan import LoanPredictionRequest

# Initialize free local embeddings globally 
try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    print(f"Warning: Could not load BAAI embeddings. {e}")

@tool
def navigate_website(target_section: str) -> str:
    """
    Use ONLY when the user explicitly asks to see work, skills, projects, or contact info.
    Valid targets: 'projects', 'skills', 'experience', 'contact'.
    """
    return f"SYSTEM_COMMAND_NAVIGATE:{target_section}"

@tool
def retrieve_portfolio_info(query: str) -> str:
    """
    Use this tool to answer questions about Janith's background, education, or work experience.
    """
    try:
        vector_store = PineconeVectorStore(index_name="portfolio-index", embedding=embeddings)
        docs = vector_store.similarity_search(query, k=2)
        context = "\n---\n".join([doc.page_content for doc in docs])
        return f"Retrieved context from Pinecone Database:\n{context}"
    except Exception as e:
        return f"Error retrieving data: {str(e)}"

@tool
async def execute_loan_prediction(
    no_of_dependents: int, education: str, self_employed: str, income_annum: float, 
    loan_amount: float, loan_term: int, cibil_score: int, residential_assets_value: float, 
    commercial_assets_value: float, luxury_assets_value: float, bank_asset_value: float
) -> str:
    """
    Use this tool to run a live machine learning inference when a user wants to check their loan eligibility.
    """
    request_data = LoanPredictionRequest(
        no_of_dependents=no_of_dependents, education=education, self_employed=self_employed,
        income_annum=income_annum, loan_amount=loan_amount, loan_term=loan_term,
        cibil_score=cibil_score, residential_assets_value=residential_assets_value,
        commercial_assets_value=commercial_assets_value, luxury_assets_value=luxury_assets_value,
        bank_asset_value=bank_asset_value
    )
    try:
        result = await predict_loan_status(request_data)
        return f"Model Prediction: {result['prediction']}"
    except Exception as e:
        return f"Error running inference: {str(e)}"