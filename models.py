from pydantic import BaseModel, Field
from typing import Dict, Any

class TransactionInput(BaseModel):
    distance_from_home: float = Field(..., description="Distance from home where the transaction happened")
    distance_from_last_transaction: float = Field(..., description="Distance from last transaction happened")
    ratio_to_median_purchase_price: float = Field(..., description="Ratio of purchased price transaction to median purchase price")
    age: int = Field(..., description="Age of the customer")
    amount: float = Field(..., description="Transaction amount")
    used_pin_number: int = Field(..., description="Is the transaction happened by using PIN number? (0 or 1)")
    online_order: int = Field(..., description="Is the transaction an online order? (0 or 1)")

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Fraud or Not Fraud")
    probability: float = Field(..., description="Probability of being fraud")
    shap_values: Dict[str, float] = Field(..., description="SHAP feature importance values")
    explanation: str = Field(..., description="LLM generated natural language explanation")
    technical_explanation: str = Field(..., description="A slightly more technical explanation")
