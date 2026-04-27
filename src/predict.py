import joblib
import pandas as pd
import numpy as np
import os

class MatchPredictor:
    def __init__(self, model_dir='models'):
        self.model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        self.feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
        self.metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
        
    def predict(self, home_stats, away_stats, custom_form=None):
        """
        home_stats: Series or dict of home team features
        away_stats: Series or dict of away team features
        custom_form: dict with overrides for form/goals
        """
        input_data = {}
        for col in self.feature_cols:
            if custom_form and col in custom_form:
                input_data[col] = custom_form[col]
            elif col.startswith('Home_'):
                # Extract the base feature name
                base_feat = col.replace('Home_', '')
                input_data[col] = home_stats.get(col, home_stats.get(f'Home_{base_feat}'))
            elif col.startswith('Away_'):
                base_feat = col.replace('Away_', '')
                input_data[col] = away_stats.get(col, away_stats.get(f'Away_{base_feat}'))
        
        input_df = pd.DataFrame([input_data])
        probs = self.model.predict_proba(input_df)[0]
        
        outcomes = ["Home Win", "Draw", "Away Win"]
        pred_idx = np.argmax(probs)
        
        return {
            'outcome': outcomes[pred_idx],
            'confidence': probs[pred_idx],
            'probabilities': dict(zip(outcomes, probs))
        }

def get_latest_team_stats(df, team_name, is_home=True):
    prefix = 'Home' if is_home else 'Away'
    team_col = 'HomeTeam' if is_home else 'AwayTeam'
    latest = df[df[team_col] == team_name].iloc[-1]
    return latest
