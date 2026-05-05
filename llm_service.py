import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

async def generate_explanation(prediction_label: str, feature_importance: dict, technical: bool = False) -> str:
    """
    Generates a natural language explanation for the prediction based on SHAP values.
    """
    
    # Sort features by absolute importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_list_str = "\n".join([f"- {k}: {v:.4f}" for k, v in sorted_features])
    
    if client:
        if technical:
            prompt = f"""
You are a technical data scientist.
Prediction: {prediction_label}
Top contributing features (SHAP values):
{feature_list_str}

Explain why this transaction is classified this way based on these SHAP values. 
Provide a concise, technically accurate explanation of how the model arrived at this decision.
            """
        else:
            prompt = f"""
You are a helpful financial fraud analyst.
Prediction: {prediction_label}
Top contributing features (SHAP values):
{feature_list_str}

Explain clearly and simply why this transaction is classified this way. 
Avoid jargon where possible, and speak as if you are explaining it to a beginner. Keep it under 4 sentences.
            """
            
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return _generate_mock_explanation(prediction_label, sorted_features, technical)
    else:
        # Mock response if no API key
        return _generate_mock_explanation(prediction_label, sorted_features, technical)

def _generate_mock_explanation(prediction_label: str, sorted_features: list, technical: bool) -> str:
    top_feature, top_val = sorted_features[0]
    second_feature, second_val = sorted_features[1]
    
    direction = "increasing" if top_val > 0 else "decreasing"
    
    if technical:
        return (f"The model predicted {prediction_label} primarily because '{top_feature}' "
                f"had a high SHAP value ({top_val:.2f}), strongly {direction} the log-odds of fraud. "
                f"This was reinforced by '{second_feature}' with a contribution of {second_val:.2f}.")
    else:
        reason = "unusually high" if top_val > 0 else "unusually low"
        action = "a warning sign for fraud" if prediction_label == "Fraud" else "typical of normal behavior"
        return (f"This transaction was flagged as {prediction_label}. The main reason is that the {top_feature.replace('_', ' ')} "
                f"was {reason}, which is {action}. Additionally, the {second_feature.replace('_', ' ')} also influenced this decision.")
