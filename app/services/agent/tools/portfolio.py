from langchain_core.tools import tool
from app.api.v1.loan import predict_loan_status
from app.schemas.loan import LoanPredictionRequest

@tool
def navigate_website(target_section: str) -> str:
    """
    Use this tool ONLY when the user explicitly asks to see your work, skills, projects, or contact info.
    Valid target_section inputs MUST be one of: 'projects', 'skills', 'experience', 'contact'.
    """
    # Returns the exact command string that the FastAPI router will intercept
    return f"SYSTEM_COMMAND_NAVIGATE:{target_section}"

@tool
def retrieve_portfolio_info(query: str) -> str:
    """
    Use this tool to answer questions about Janith's background, education, or work experience.
    """
    # This acts as a hardcoded RAG retriever to prevent AI hallucinations.
    context = (
        "Janith Gayashan is an AI Engineer and student pursuing a Bachelor of Science (Hons) "
        "in Artificial Intelligence at the University of Moratuwa. "
        "He completed a five-month internship at SLT Digital Lab contributing to a real-world "
        "AI project called the slt-travel-chatbot. "
        "He has experience building reactive multi-agent nodes using LangGraph, enterprise-level "
        "RAG platforms, and deploying Machine Learning pipelines using FastAPI and Scikit-Learn."
    )
    return context

@tool
async def execute_loan_prediction(
    no_of_dependents: int, education: str, self_employed: str, income_annum: float, 
    loan_amount: float, loan_term: int, cibil_score: int, residential_assets_value: float, 
    commercial_assets_value: float, luxury_assets_value: float, bank_asset_value: float
) -> str:
    """
    Use this tool to run a live machine learning inference when a user wants to check their loan eligibility.
    You must ask the user for all of these financial parameters before running this tool.
    """
    # 1. Package the agent's parameters into the Pydantic model your API expects
    request_data = LoanPredictionRequest(
        no_of_dependents=no_of_dependents, education=education, self_employed=self_employed,
        income_annum=income_annum, loan_amount=loan_amount, loan_term=loan_term,
        cibil_score=cibil_score, residential_assets_value=residential_assets_value,
        commercial_assets_value=commercial_assets_value, luxury_assets_value=luxury_assets_value,
        bank_asset_value=bank_asset_value
    )
    
    # 2. Directly trigger your existing Decision Tree logic
    try:
        result = await predict_loan_status(request_data)
        return f"The Machine Learning model classified this applicant as: {result['prediction']}"
    except Exception as e:
        return f"Error running inference: {str(e)}"