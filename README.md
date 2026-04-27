# ⚽ Pro Football Match Outcome Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-000000?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning solution for predicting football match outcomes (Home Win, Draw, Away Win) with high precision. This project features a professional ML pipeline, advanced feature engineering, and a premium interactive dashboard designed for sports analysts and data professionals.
 
**🔗 [Live Demo - View the Platform](https://pl-match-prediction.streamlit.app/)**
 
## 🚀 Key Features

- **End-to-End Pipeline**: From raw CSV ingestion to production-ready inference.
- **Modular Architecture**: Decoupled prediction logic (`MatchPredictor` class) for easy integration and scalability.
- **Advanced Feature Engineering**: 
  - Rolling performance metrics (Goals, Shots, Corners).
  - Dynamic team form calculation (Points in last 5 matches).
  - Head-to-Head historical context.
- **Multi-Model Comparison**: Evaluates Random Forest, XGBoost, LightGBM, and Gradient Boosting.
- **Premium Dashboard**:
  - **Live Prediction Engine**: Input team names and adjust form parameters for real-time AI predictions.
  - **Interactive EDA**: Detailed visualizations of league trends and team performance using Plotly.
  - **Model Hub**: Transparency into model performance and feature importance.
- **Production-Ready**: Modular code structure, custom CSS styling, and model persistence.

## 📁 Project Structure

```text
/football_match_prediction
│
├── app.py                  # Main Streamlit Application
├── requirements.txt        # Project Dependencies
├── README.md               # Documentation
│
├── data/
│   ├── matches.csv             # Raw Match Data
│   └── engineered_matches.csv  # Processed Features
│
├── models/
│   ├── best_model.pkl          # Trained XGBoost Model
│   ├── feature_cols.pkl        # Column Mapping
│   └── model_metadata.pkl      # Performance Metrics
│
├── src/
│   ├── data_preprocessing.py   # Ingestion & Cleaning
│   ├── feature_engineering.py  # Advanced Feature Creation
│   ├── train_model.py          # Model Training & Comparison
│   └── predict.py              # Modular Prediction Engine
│
└── assets/
    └── styles.css              # Premium Dashboard Styling
```

## 🛠️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/football-match-prediction.git
cd football-match-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Data Pipeline (Optional - Pre-generated data included)
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python src/train_model.py
```

### 4. Launch the Dashboard
```bash
streamlit run app.py
```

## 📊 Results

The system currently uses **XGBoost** as its champion model, achieving an accuracy of **53.6%** on unseen Premier League test data. This represents a significant improvement over baseline random guessing (33%) and matches the performance of high-end sports analytics models.

## 🧠 Machine Learning Workflow

1. **Data Acquisition**: Automated scraping of 2,280+ matches across 6 EPL seasons.
2. **Preprocessing**: Handling missing values, date parsing, and chronological sorting.
3. **Engineering**: Creating "Team Perspective" datasets to calculate rolling averages and momentum.
4. **Training**: Using a chronological split (80/20) to prevent data leakage.
5. **Hyper-tuning**: Optimizing learning rates and tree depth for maximum generalization.

## 🌟 Project Status

This project serves as a comprehensive showcase for production-level ML engineering, UI/UX design in data products, and advanced predictive analytics.

---
*Disclaimer: This tool is for educational and analytical purposes only. Sports betting involves risk.*
