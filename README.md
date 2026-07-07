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

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows
source .venv/bin/activate         # Mac/Linux
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
- `pool` — swimming pool availability (0 or 1)
- `basement` — basement availability (0 or 1)

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
- Location: Mumbai
- Age: 10 years
- Garage: 2 spaces
- Pool: No
- Basement: Yes

Output:
- Predicted Price: ₹1,25,00,000

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

