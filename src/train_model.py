import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_and_split_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Define features and target
    # We exclude Date, Team names, and the raw FTR
    feature_cols = [c for c in df.columns if 'Rolling' in c or 'FormPoints' in c]
    X = df[feature_cols]
    y = df['Target']
    
    # Chronological split (e.g., last 20% for testing)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_acc:.4f}")
    return best_model, best_model_name, results

def main():
    try:
        data_path = 'data/engineered_matches.csv'
        X_train, X_test, y_train, y_test, feature_cols = load_and_split_data(data_path)
        
        best_model, best_name, results = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/best_model.pkl'
        joblib.dump(best_model, model_path)
        
        # Save feature columns to ensure consistency in app
        joblib.dump(feature_cols, 'models/feature_cols.pkl')
        
        # Save metadata for the app
        metadata = {
            'best_model_name': best_name,
            'accuracy': results[best_name],
            'all_results': results
        }
        joblib.dump(metadata, 'models/model_metadata.pkl')
        
        print(f"Model saved to {model_path}")
        
    except FileNotFoundError:
        print("data/engineered_matches.csv not found. Run feature_engineering.py first.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()
