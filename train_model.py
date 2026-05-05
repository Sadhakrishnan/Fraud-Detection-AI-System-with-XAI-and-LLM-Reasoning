import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_dummy_data(n_samples=5000):
    """Generates synthetic dataset for credit card fraud detection."""
    print("Generating synthetic dataset...")
    
    # Feature 1: distance_from_home
    distance_from_home = np.random.exponential(scale=50, size=n_samples)
    
    # Feature 2: distance_from_last_transaction
    distance_from_last_transaction = np.random.exponential(scale=10, size=n_samples)
    
    # Feature 3: ratio_to_median_purchase_price
    ratio_to_median_purchase_price = np.random.lognormal(mean=0, sigma=1, size=n_samples)
    
    # Feature 4: age
    age = np.random.randint(18, 80, size=n_samples)
    
    # Feature 5: amount
    amount = np.random.exponential(scale=100, size=n_samples)
    
    # Feature 6: used_pin_number (Boolean)
    used_pin_number = np.random.binomial(1, 0.5, size=n_samples)
    
    # Feature 7: online_order (Boolean)
    online_order = np.random.binomial(1, 0.6, size=n_samples)
    
    # Target variable: Fraud
    # Fraud is more likely if:
    # 1. High ratio to median purchase price
    # 2. High distance from home
    # 3. Online order
    # 4. Didn't use PIN
    
    # Calculate a raw "fraud score" based on some non-linear combination
    fraud_score = (
        (ratio_to_median_purchase_price * 2) +
        (distance_from_home / 50) +
        (amount / 200) +
        (online_order * 3) -
        (used_pin_number * 5)
    )
    
    # Convert to probability and then to binary class (imbalanced)
    # Target ~5-10% fraud rate
    prob_fraud = 1 / (1 + np.exp(-(fraud_score - 8))) 
    fraud = np.random.binomial(1, prob_fraud)
    
    df = pd.DataFrame({
        'distance_from_home': distance_from_home,
        'distance_from_last_transaction': distance_from_last_transaction,
        'ratio_to_median_purchase_price': ratio_to_median_purchase_price,
        'age': age,
        'amount': amount,
        'used_pin_number': used_pin_number,
        'online_order': online_order,
        'fraud': fraud
    })
    
    print(f"Dataset created with {n_samples} samples. Fraud cases: {df['fraud'].sum()} ({df['fraud'].mean()*100:.2f}%)")
    return df

def train_and_save_model():
    df = generate_dummy_data(10000)
    
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Calculate scale_pos_weight for class imbalance
    neg_cases = (y_train == 0).sum()
    pos_cases = (y_train == 1).sum()
    scale_pos_weight = neg_cases / pos_cases if pos_cases > 0 else 1.0
    
    print(f"Training XGBoost model (scale_pos_weight={scale_pos_weight:.2f})...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save model and feature names
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'xgboost_fraud_model.joblib')
    
    # Save a dictionary containing the model and the feature names to ensure order is preserved
    joblib.dump({
        'model': model,
        'features': list(X.columns)
    }, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
