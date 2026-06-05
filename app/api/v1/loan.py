from fastapi import APIRouter, HTTPException
from app.schemas.loan import LoanPredictionRequest, LoanPredictionResponse
import pandas as pd
import joblib
import os

router = APIRouter()

# This is use for Load models globally so they don't reload on every single API request
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "loan_decision_tree_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml_models", "loan_scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Warning: ML models not found at {MODEL_PATH}. Error: {e}")

@router.post("/predict", response_model=LoanPredictionResponse)
async def predict_loan_status(request: LoanPredictionRequest):
    try:
        # 1. Label Encode the categorical variables
        education_mapped = 1 if request.education.strip().lower() == 'not graduate' else 0
        self_employed_mapped = 1 if request.self_employed.strip().lower() == 'yes' else 0
        
        # 2. Assemble the exact feature DataFrame your model expects (Silences the warning)
        features = pd.DataFrame([[
            request.no_of_dependents,
            education_mapped,
            self_employed_mapped,
            request.income_annum,
            request.loan_amount,
            request.loan_term,
            request.cibil_score,
            request.residential_assets_value,
            request.commercial_assets_value,
            request.luxury_assets_value,
            request.bank_asset_value
        ]], columns=[
            "no_of_dependents", "education", "self_employed", "income_annum",
            "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
            "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
        ])
        
        # 3. Apply the StandardScaler
        scaled_features = scaler.transform(features)
        
        # 4. Execute the Decision Tree inference
        prediction = model.predict(scaled_features)
        
        # 5. Decode the prediction: 0 is Approved, 1 is Rejected (Alphabetical mapping fix)
        result = "Approved" if prediction[0] == 0 else "Rejected"
        
        return {"prediction": result, "status_code": 200}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))