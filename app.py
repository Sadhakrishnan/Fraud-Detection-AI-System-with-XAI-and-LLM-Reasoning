import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as express
import pandas as pd

# Set page config for premium look
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode premium aesthetics
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Form Styling */
    .stNumberInput > div > div > input {
        background-color: #1E2329 !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3F45;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.4);
        color: white;
    }
    
    /* Panels */
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    
    /* Cards */
    .metric-card {
        background-color: #1E2329;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3A3F45;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.title("🛡️ Next-Gen Fraud Detection System")
st.markdown("Enter transaction details below to get an AI-powered risk assessment with complete explainability.")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Transaction Details")
    with st.form("transaction_form"):
        distance_from_home = st.number_input("Distance from Home (miles)", min_value=0.0, value=5.0, step=1.0)
        distance_from_last_transaction = st.number_input("Distance from Last Transaction (miles)", min_value=0.0, value=2.0, step=1.0)
        ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, value=1.2, step=0.1)
        age = st.number_input("Customer Age", min_value=18, max_value=120, value=35, step=1)
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=5.0)
        
        used_pin_number = st.selectbox("Used PIN Number?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        online_order = st.selectbox("Online Order?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        submit_button = st.form_submit_button(label="Analyze Transaction")

with col2:
    if submit_button:
        with st.spinner("Analyzing transaction patterns with AI..."):
            payload = {
                "distance_from_home": distance_from_home,
                "distance_from_last_transaction": distance_from_last_transaction,
                "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
                "age": age,
                "amount": amount,
                "used_pin_number": used_pin_number,
                "online_order": online_order
            }
            
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Setup metrics layout
                st.markdown("### Analysis Results")
                
                pred_color = "#EF4444" if result["prediction"] == "Fraud" else "#10B981"
                
                # Gauge Chart for Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result["probability"] * 100,
                    title = {'text': "Fraud Probability (%)", 'font': {'color': 'white'}},
                    number = {'font': {'color': pred_color}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': pred_color},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    height=250,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                st.markdown(f"<h2 style='text-align: center; color: {pred_color};'>Verdict: {result['prediction']}</h2>", unsafe_allow_html=True)
                
                st.divider()
                
                # Explanation Section
                st.markdown("### AI Reasoning")
                
                exp_tabs = st.tabs(["Simple Explanation", "Technical Deep Dive"])
                
                with exp_tabs[0]:
                    st.info(result["explanation"])
                    
                with exp_tabs[1]:
                    st.warning(result["technical_explanation"])
                
                st.divider()
                
                # SHAP Feature Importance Chart
                st.markdown("### Feature Importance (SHAP)")
                
                shap_values = result["shap_values"]
                shap_df = pd.DataFrame(list(shap_values.items()), columns=["Feature", "Impact"])
                shap_df["Color"] = shap_df["Impact"].apply(lambda x: "#EF4444" if x > 0 else "#10B981")
                # Sort by absolute impact
                shap_df["AbsImpact"] = shap_df["Impact"].abs()
                shap_df = shap_df.sort_values(by="AbsImpact", ascending=True)
                
                fig_bar = express.bar(
                    shap_df, 
                    x="Impact", 
                    y="Feature", 
                    orientation='h',
                    title="How each feature contributed to the score",
                )
                
                fig_bar.update_traces(marker_color=shap_df["Color"])
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                fig_bar.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='white')
                fig_bar.update_yaxes(showgrid=False)
                
                st.plotly_chart(fig_bar, use_container_width=True)

            except requests.exceptions.ConnectionError:
                st.error("⚠️ Cannot connect to the backend server. Is it running?")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
    else:
        st.info("👈 Enter transaction details and click 'Analyze Transaction' to see the results.")
