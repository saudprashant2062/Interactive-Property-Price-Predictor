# Tech Stack Analysis: Property Price Predictor

This project is a __machine learning web application__ that predicts residential property prices. Here's a comprehensive breakdown of the technology stack:

## 🐍 __Core Language & Runtime__

- __Python__ - The entire project is built in Python, a popular language for data science and machine learning

## 📊 __Data Processing & Analysis__

- __Pandas (>=2.2.0)__ - Data manipulation and analysis library. Used for:

  - Loading CSV data
  - Creating DataFrames for predictions
  - Data transformation and feature engineering

- __NumPy (>=1.26.0)__ - Numerical computing library. Used for:

  - Mathematical operations
  - Array manipulations
  - Statistical calculations (mean, std, etc.)

## 🤖 __Machine Learning__

- __XGBoost (>=2.0.0)__ - Gradient boosting framework. The core ML algorithm:

  - `XGBRegressor` for price prediction
  - Hyperparameters: 200 estimators, learning rate 0.05, max depth 6
  - Provides feature importance analysis

- __Scikit-learn (>=1.5.0)__ - Machine learning library for:

  - `StandardScaler` - Feature scaling/normalization
  - `OneHotEncoder` - Categorical variable encoding
  - `train_test_split` - Data splitting
  - Model evaluation metrics (MAE, RMSE, R²)
  - Cross-validation (`cross_val_score`)

## 🖥️ __Web Framework & UI__

- __Streamlit (>=1.40.0)__ - Web application framework for creating the interactive UI:

  - Interactive sliders, select boxes, and checkboxes
  - Real-time predictions
  - Model retraining capability
  - Responsive layout with columns and sidebars

## 📈 __Data Visualization__

- __Plotly (>=5.24.0)__ - Interactive graphing library:

  - Feature importance bar charts
  - Interactive visualizations in the Streamlit app

- __Matplotlib (>=3.8.0)__ & __Seaborn (>=0.13.0)__ - Static plotting libraries (included but not actively used in current code)

## 💾 __Model Persistence__

- __Joblib (>=1.4.0)__ - Model serialization library:

  - Saving/training models (`dump`)
  - Loading trained models (`load`)
  - More efficient than pickle for numpy arrays

## 🏗️ __Project Architecture__

```javascript
property-price-predictor/
├── app/                    # Streamlit web application
│   └── app.py             # Main UI and prediction logic
├── src/                    # Core ML and preprocessing
│   ├── config.py          # Configuration and paths
│   ├── preprocessing.py   # Data cleaning and transformation
│   ├── train.py           # Model training pipeline
│   └── sample_data.py     # Synthetic data generation
├── models/                 # Trained model artifacts
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   └── encoder.joblib
├── data/                   # Dataset storage
│   ├── raw/               # Input data (properties.csv)
│   └── processed/         # Processed data
└── requirements.txt       # Python dependencies
```

## 🔧 __Key Technical Features__

1. __Preprocessing Pipeline:__

   - Outlier detection (IQR and Z-score methods)
   - Missing value handling
   - Feature scaling (StandardScaler)
   - Categorical encoding (OneHotEncoder)

2. __Model Training:__

   - 80-20 train-test split
   - 5-fold cross-validation
   - Evaluation metrics: MAE, RMSE, R² score

3. __Prediction System:__

   - Real-time price estimation
   - Feature importance visualization
   - Model retraining capability

## 📋 __Summary__

This is a __modern Python ML stack__ combining:

- __Data Science__: Pandas, NumPy
- __Machine Learning__: XGBoost, Scikit-learn
- __Web Development__: Streamlit
- __Visualization__: Plotly
- __Model Management__: Joblib

The architecture follows best practices with separation of concerns (data processing, training, and web app in different modules) and includes features like automatic sample data generation, model persistence, and interactive visualizations.


# Interactive Property Price Predictor

A machine learning web application that predicts residential property prices using XGBoost and Streamlit.

## Project Highlights

- **XGBoost Regression Model** — predicts prices based on property features
- **Automated Preprocessing** — handles outlier detection, feature scaling, and categorical encoding
- **Interactive Streamlit App** — real-time predictions with user-friendly interface
- **Feature Engineering** — creates meaningful features from raw property data
- **Model Evaluation** — MAE, RMSE, R² score metrics

## Folder Structure

- `src/` — preprocessing and model training code
- `app/` — Streamlit application
- `models/` — trained XGBoost models
- `data/` — raw and processed datasets
- `notebooks/` — exploratory analysis (optional)

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows
source .venv/bin/activate        # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Place your dataset at `data/raw/properties.csv` with columns:
- `price` — target variable (house price)
- `sqft` — square footage
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `location` — neighborhood or zip code
- `age` — house age in years
- `garage` — number of garage spaces

If no data is provided, the app will generate a synthetic dataset automatically.

### 4. Train Model (Optional)

```bash
python -m src.train
```

### 5. Launch Streamlit App

```bash
streamlit run app/app.py
```

The app will open at `http://localhost:8501`

## Features

### Data Preprocessing

- Outlier detection and removal
- Missing value handling
- Feature scaling (StandardScaler)
- Categorical variable encoding (One-Hot)
- Feature engineering and selection

### Model

- **Algorithm** — XGBoost Regression
- **Hyperparameters** — tuned for property prediction
- **Cross-validation** — 5-fold for robustness
- **Feature Importance** — visualized in the app

### Interactive Predictions

1. Input property features using sliders and selectors
2. Get instant price prediction
3. View model confidence and feature importance
4. Explore historical predictions

## Example Usage

Input:
- Square footage: 2500 sqft
- Bedrooms: 4
- Bathrooms: 2.5
- Location: Downtown
- Age: 10 years
- Garage: 2 spaces

Output:
- Predicted Price: $450,000
- Model Confidence: 92%
- Price Range: $420,000 - $480,000

## Performance Metrics

The model is evaluated on:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R² Score** — Coefficient of Determination

## Notes

- The synthetic dataset is for demo purposes only
- For production, use real property market data
- Retrain the model periodically with fresh data

## License

MIT

