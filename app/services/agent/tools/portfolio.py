import os
import logging
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.api.v1.loan import predict_loan_status
from app.schemas.loan import LoanPredictionRequest

# Initialize module-level logger
logger = logging.getLogger(__name__)

# Initialize free local embeddings globally 
try:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    logger.error(f"❌ EMBEDDINGS ERROR: Could not load BAAI embeddings. Details: {e}")

@tool
def navigate_website(target_section: str) -> str:
    """
    Use ONLY when the user explicitly asks to see work, skills, projects, or contact info.
    Valid targets: 'projects', 'skills', 'experience', 'contact'.
    """
    logger.info(f"🌐 TOOL ACTIVATED [navigate_website] -> Intercepted request for client-side navigation to: '{target_section}'")
    return f"SYSTEM_COMMAND_NAVIGATE:{target_section}"

@tool
def retrieve_portfolio_info(query: str) -> str:
    """Use this tool to answer questions about Janith's background, education, or work experience."""
    logger.info(f"🗄️ TOOL ACTIVATED [retrieve_portfolio_info] -> Preparing vector search on Pinecone for query: '{query}'")
    try:
        vector_store = PineconeVectorStore(index_name="portfolio-index", embedding=embeddings)
        
        # 1. Fetch documents along with similarity scores
        logger.info("🔎 PINECONE: Querying vector index for similarity matches...")
        docs_with_scores = vector_store.similarity_search_with_score(query, k=2)
        
        # 2. Enforce a strict confidence score threshold (e.g., Cosine similarity > 0.75)
        valid_docs = [doc for doc, score in docs_with_scores if score >= 0.75]
        
        if not valid_docs:
            logger.warning(f"⚠️ PINECONE: Query finished. 0 documents satisfied the strict similarity threshold of >= 0.75.")
            return "GUARDRAIL_TRIGGERED: The database does not contain any verified information matching this specific query. Explicitly state that you do not have access to this information."
            
        logger.info(f"✅ PINECONE: Found {len(valid_docs)} document(s) matching threshold parameters. Constructing prompt context...")
        context = "\n---\n".join([doc.page_content for doc in valid_docs])
        return f"Retrieved context from Pinecone Database:\n{context}"
        
    except Exception as e:
        logger.error(f"❌ PINECONE ERROR: Exception encountered during index search processing. Trace: {str(e)}")
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
    logger.info(f"🧠 TOOL ACTIVATED [execute_loan_prediction] -> Building Pydantic request payload. Client CIBIL Score: {cibil_score}")
    request_data = LoanPredictionRequest(
        no_of_dependents=no_of_dependents, education=education, self_employed=self_employed,
        income_annum=income_annum, loan_amount=loan_amount, loan_term=loan_term,
        cibil_score=cibil_score, residential_assets_value=residential_assets_value,
        commercial_assets_value=commercial_assets_value, luxury_assets_value=luxury_assets_value,
        bank_asset_value=bank_asset_value
    )
    try:
        logger.info("📈 INFERENCE: Piping feature space metrics down to local scikit-learn/XGBoost prediction service...")
        result = await predict_loan_status(request_data)
        logger.info(f"✨ INFERENCE SUCCESS: Prediction successfully resolved to -> Status: {result.get('prediction')}")
        return f"Model Prediction: {result['prediction']}"
    except Exception as e:
        logger.error(f"❌ INFERENCE ERROR: Pipeline evaluation broke during feature computation. Stack trace: {str(e)}")
        return f"Error running inference: {str(e)}"