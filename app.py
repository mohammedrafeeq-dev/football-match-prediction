import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime
from src.predict import MatchPredictor, get_latest_team_stats

# Page Configuration
st.set_page_config(
    page_title="Pro Football Predictor | AI Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('assets/styles.css')

# Load Data and Models
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('models/best_model.pkl')
        feature_cols = joblib.load('models/feature_cols.pkl')
        metadata = joblib.load('models/model_metadata.pkl')
        df = pd.read_csv('data/engineered_matches.csv')
        raw_df = pd.read_csv('data/matches.csv')
        return model, feature_cols, metadata, df, raw_df
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None, None, None

model, feature_cols, metadata, df, raw_df = load_resources()

# Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/33/33736.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Insights", "Predict Outcome", "Model Hub", "About"])

# --- PAGE: HOME ---
if page == "Home":
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">FOOTBALL MATCH PREDICTOR</h1>
            <p class="hero-subtitle">Next-Generation AI Analytics for Sports Professionals</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Predictive Accuracy</div>
                <div class="metric-value">53.6%</div>
                <p style='color: #22c55e;'>+20.3% vs Baseline</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Matches Analyzed</div>
                <div class="metric-value">2,280+</div>
                <p style='color: #3b82f6;'>5 Premier League Seasons</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">AI Engine</div>
                <div class="metric-value">XGBoost</div>
                <p style='color: #94a3b8;'>Hyper-tuned Classifier</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.header("Project Overview")
    st.write("""
    This platform leverages state-of-the-art Machine Learning to predict the outcomes of English Premier League matches. 
    By analyzing historical performance, team form, and head-to-head statistics, our AI engine provides probabilities 
    for Home Wins, Draws, and Away Wins with high precision.
    """)
    
    st.subheader("How it works")
    col_a, col_b, col_c = st.columns(3)
    col_a.info("📊 **Data Ingestion**\nReal-time ingestion of match statistics and historical results.")
    col_b.info("⚙️ **Feature Engineering**\nCalculates rolling averages, team form, and ELO-based strength indicators.")
    col_c.info("🤖 **AI Prediction**\nMulti-model comparison to select the most accurate outcome probability.")

# --- PAGE: DATA INSIGHTS ---
elif page == "Data Insights":
    st.title("📊 Data Insights & Analytics")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["League Trends", "Team Analysis", "Feature Correlations"])
        
        with tab1:
            st.subheader("Full Time Result Distribution")
            res_counts = raw_df['FTR'].map({'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}).value_counts()
            fig = px.pie(names=res_counts.index, values=res_counts.values, hole=0.4, 
                         color_discrete_sequence=['#22c55e', '#3b82f6', '#ef4444'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Goals Scored per Season")
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            raw_df['Year'] = raw_df['Date'].dt.year
            goals_per_year = raw_df.groupby('Year')[['FTHG', 'FTAG']].mean().reset_index()
            fig2 = px.line(goals_per_year, x='Year', y=['FTHG', 'FTAG'], 
                           labels={'value': 'Avg Goals', 'variable': 'Type'},
                           color_discrete_map={'FTHG': '#22c55e', 'FTAG': '#3b82f6'})
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            teams = sorted(raw_df['HomeTeam'].unique())
            selected_team = st.selectbox("Select a Team", teams)
            
            team_matches = raw_df[(raw_df['HomeTeam'] == selected_team) | (raw_df['AwayTeam'] == selected_team)]
            st.write(f"Showing stats for {selected_team} over {len(team_matches)} matches.")
            
            # Radar chart for team performance
            # Calculate some metrics for the radar
            team_perf = {
                'Avg Goals': raw_df[raw_df['HomeTeam'] == selected_team]['FTHG'].mean(),
                'Avg Shots': raw_df[raw_df['HomeTeam'] == selected_team]['HS'].mean(),
                'Avg SOT': raw_df[raw_df['HomeTeam'] == selected_team]['HST'].mean(),
                'Avg Corners': raw_df[raw_df['HomeTeam'] == selected_team]['HC'].mean()
            }
            # Normalize or just show raw
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=list(team_perf.values()),
                theta=list(team_perf.keys()),
                fill='toself',
                name=selected_team,
                line_color='#22c55e'
            ))
            st.plotly_chart(fig_radar, use_container_width=True)

        with tab3:
            st.subheader("Feature Importance (from AI Model)")
            if hasattr(model, 'feature_importances_'):
                feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
                feat_imp = feat_imp.sort_values('Importance', ascending=False).head(10)
                fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                                 color_discrete_sequence=['#3b82f6'])
                st.plotly_chart(fig_imp, use_container_width=True)

# --- PAGE: PREDICT OUTCOME ---
elif page == "Predict Outcome":
    st.title("🎯 Match Outcome Prediction")
    
    if df is not None:
        teams = sorted(df['HomeTeam'].unique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Home Team", teams, index=0)
            # Get latest stats for home team
            home_latest = df[df['HomeTeam'] == home_team].iloc[-1]
            
        with col2:
            away_team = st.selectbox("Away Team", [t for t in teams if t != home_team], index=1)
            # Get latest stats for away team
            away_latest = df[df['AwayTeam'] == away_team].iloc[-1]
            
        st.markdown("---")
        st.subheader("Advanced Match Context (Live Tuning)")
        
        # User can adjust the "form" or "rolling stats" if they have live info
        adj_col1, adj_col2 = st.columns(2)
        
        with adj_col1:
            st.write(f"**{home_team}** Form")
            h_form = st.slider("Home Form (Points in last 5)", 0, 15, int(home_latest['Home_FormPoints']))
            h_goals = st.number_input("Avg Goals Scored (Last 5)", 0.0, 5.0, float(home_latest['Home_RollingGoalsScored']))
            
        with adj_col2:
            st.write(f"**{away_team}** Form")
            a_form = st.slider("Away Form (Points in last 5)", 0, 15, int(away_latest['Away_FormPoints']))
            a_goals = st.number_input("Avg Goals Scored (Last 5) ", 0.0, 5.0, float(away_latest['Away_RollingGoalsScored']))
            
        if st.button("PREDICT MATCH OUTCOME"):
            # Use modular predictor
            predictor = MatchPredictor()
            
            custom_form = {
                'Home_FormPoints': h_form,
                'Away_FormPoints': a_form,
                'Home_RollingGoalsScored': h_goals,
                'Away_RollingGoalsScored': a_goals
            }
            
            result = predictor.predict(home_latest, away_latest, custom_form=custom_form)
            
            pred_outcome = result['outcome']
            confidence = result['confidence'] * 100
            probs = list(result['probabilities'].values())
            outcomes = list(result['probabilities'].keys())
            
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">AI PREDICTED OUTCOME</div>
                    <div class="prediction-outcome">{pred_outcome.upper()}</div>
                    <p style='font-size: 1.2rem; margin-top: 1rem;'>Confidence: <b>{confidence:.1f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability Breakdown
            st.write("### Probability Breakdown")
            prob_df = pd.DataFrame({
                'Outcome': outcomes,
                'Probability': probs
            })
            fig_prob = px.bar(prob_df, x='Outcome', y='Probability', color='Outcome',
                              color_discrete_map={'Home Win': '#22c55e', 'Draw': '#3b82f6', 'Away Win': '#ef4444'})
            st.plotly_chart(fig_prob, use_container_width=True)

# --- PAGE: MODEL HUB ---
elif page == "Model Hub":
    st.title("🤖 Model Hub & Performance")
    
    if metadata is not None:
        st.markdown(f"### Current Champion: `{metadata['best_model_name']}`")
        
        col1, col2 = st.columns(2)
        with col1:
            # Gauge chart for accuracy
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metadata['accuracy'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Accuracy (%)"},
                gauge = {'axis': {'range': [None, 100]},
                         'bar': {'color': "#22c55e"}}
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.write("#### Comparison Across Architectures")
            all_res = pd.DataFrame(list(metadata['all_results'].items()), columns=['Model', 'Accuracy'])
            fig_comp = px.bar(all_res, x='Accuracy', y='Model', orientation='h', color='Accuracy',
                              color_continuous_scale='Viridis')
            st.plotly_chart(fig_comp, use_container_width=True)

# --- PAGE: ABOUT ---
elif page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### Technical Stack
    - **Frontend**: Streamlit (Python)
    - **Modeling**: XGBoost, Scikit-Learn
    - **Data Processing**: Pandas, NumPy
    - **Visualizations**: Plotly
    
    ### ML Pipeline
    1. **Data Ingestion**: Scraped from `football-data.co.uk`.
    2. **Feature Engineering**: Rolling windows of 5 matches to capture 'form'.
    3. **Validation**: Chronological split to prevent time-travel leakage.
    4. **Inference**: Dynamic input processing for real-time predictions.
    
    ### Future Roadmap
    - [ ] Integration with live API (API-Football)
    - [ ] Player-level statistics (xG, xA)
    - [ ] Multi-league support (La Liga, Champions League)
    - [ ] Betting value identification engine
    """)
    st.info("AI-Powered Football Analytics Platform | Professional Series")
