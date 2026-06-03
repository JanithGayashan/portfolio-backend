from pydantic import BaseModel, Field

class LoanPredictionRequest(BaseModel):
    no_of_dependents: int
    education: str = Field(..., description="'Graduate' or 'Not Graduate'")
    self_employed: str = Field(..., description="'Yes' or 'No'")
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

class LoanPredictionResponse(BaseModel):
    prediction: str
    status_code: int