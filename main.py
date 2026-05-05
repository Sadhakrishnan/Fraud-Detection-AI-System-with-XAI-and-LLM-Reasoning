import os
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import TransactionInput, PredictResponse
from llm_service import generate_explanation

app = FastAPI(title="Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
model_dict = None
model = None
feature_names = None
explainer = None

@app.on_event("startup")
async def load_model():
    global model_dict, model, feature_names, explainer
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_fraud_model.joblib')
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Please run train_model.py first.")
    
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    feature_names = model_dict['features']
    
    # Initialize SHAP explainer
    # TreeExplainer is efficient for XGBoost
    explainer = shap.TreeExplainer(model)
    print("Model and explainer loaded successfully.")

@app.post("/predict", response_model=PredictResponse)
async def predict(transaction: TransactionInput):
    if not model or not explainer:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    
    try:
        # Convert input to DataFrame with the exact feature order used during training
        input_data = transaction.dict()
        df = pd.DataFrame([input_data])[feature_names]
        
        # Predict
        prob = model.predict_proba(df)[0, 1]
        is_fraud = bool(prob > 0.5) # Using 0.5 threshold, can be adjusted
        prediction_label = "Fraud" if is_fraud else "Not Fraud"
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(df)
        
        # If output has multiple dimensions (e.g. for multi-class or some setups), take the class 1 output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        shap_dict = {feature_names[i]: float(shap_values[0][i]) for i in range(len(feature_names))}
        
        # Generate explanations (concurrently if we wanted, but sequential is fine for now)
        explanation = await generate_explanation(prediction_label, shap_dict, technical=False)
        tech_explanation = await generate_explanation(prediction_label, shap_dict, technical=True)
        
        return PredictResponse(
            prediction=prediction_label,
            probability=float(prob),
            shap_values=shap_dict,
            explanation=explanation,
            technical_explanation=tech_explanation
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
