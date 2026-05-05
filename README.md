# Fraud Detection AI System

This project is an end-to-end fraud detection system that combines machine learning, explainability, and a simple web interface. It predicts whether a transaction is fraudulent and explains the reasoning behind the prediction.

## Overview

The system includes:

* A machine learning model trained on synthetic transaction data
* A FastAPI backend for serving predictions
* SHAP-based explanations for model decisions
* A Streamlit frontend for user interaction

## Features

* Predicts fraud probability for transactions
* Handles imbalanced data using XGBoost
* Provides feature-level explanations using SHAP
* Generates human-readable explanations (with or without OpenAI API)
* Interactive dashboard for testing transactions

## Project Structure

.
├── train_model.py       # Data generation and model training
├── main.py              # FastAPI backend
├── llm_service.py       # Explanation logic
├── app.py               # Streamlit frontend
├── model.joblib         # Saved model
├── requirements.txt
└── .env                 # Optional API key
