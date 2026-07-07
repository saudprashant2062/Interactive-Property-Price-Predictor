import logging
import streamlit as st
from pathlib import Path
import pandas as pd
from joblib import load
import plotly.graph_objects as go
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEFAULT_MODEL_FILE,
    DEFAULT_SCALER_FILE,
    DEFAULT_ENCODER_FILE,
    DEFAULT_INPUT_FILE,
)
from src.sample_data import generate_sample_properties
from src.train import train as train_model
from src.preprocessing import scale_numeric_features, encode_categorical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Property Price Predictor", layout="wide", initial_sidebar_state="expanded")

LOCATIONS = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune"]


def load_models():
    try:
        if not DEFAULT_MODEL_FILE.exists():
            st.warning("Model not found. Training model on demo data...")
            if not DEFAULT_INPUT_FILE.exists():
                generate_sample_properties(DEFAULT_INPUT_FILE)
            train_model()

        model = load(DEFAULT_MODEL_FILE)
        scaler = load(DEFAULT_SCALER_FILE)
        encoder = load(DEFAULT_ENCODER_FILE)
        return model, scaler, encoder
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error(f"Failed to load model: {e}")
        st.stop()


def create_prediction_df(sqft, bedrooms, bathrooms, location, age, garage, pool, basement, scaler, encoder):
    input_data = pd.DataFrame({
        'sqft': [sqft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'location': [location],
        'age': [age],
        'garage': [garage],
        'pool': [pool],
        'basement': [basement],
    })

    input_data, _ = scale_numeric_features(input_data, scaler, fit=False)
    input_data, _ = encode_categorical_features(input_data, encoder, fit=False)

    return input_data


def main():
    st.title("Interactive Property Price Predictor")
    st.markdown("Predict residential property prices using machine learning (XGBoost)")

    try:
        model, scaler, encoder = load_models()
    except Exception:
        return

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Property Details")
        sqft = st.slider("Square Footage", 500, 10000, 2500, 100)
        bedrooms = st.slider("Bedrooms", 1, 10, 3, 1)
        bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.0, 0.5)
        location = st.selectbox("Location", LOCATIONS)
        age = st.slider("Age (Years)", 0, 100, 10, 1)
        garage = st.slider("Garage Spaces", 0, 4, 1, 1)
        pool = st.checkbox("Swimming Pool Available", value=False)
        basement = st.checkbox("Basement Available", value=False)

    with col2:
        st.header("Price Prediction")

        try:
            input_df = create_prediction_df(
                sqft, bedrooms, bathrooms, location, age, garage,
                int(pool), int(basement), scaler, encoder
            )

            predicted_price = model.predict(input_df)[0]

            col_price, _ = st.columns([2, 1])
            with col_price:
                st.metric("Estimated Price", f"₹{predicted_price:,.0f}")

            st.markdown("---")
            st.subheader("Property Summary")
            summary_data = {
                "Square Footage": f"{sqft:,} sqft",
                "Bedrooms": str(bedrooms),
                "Bathrooms": str(bathrooms),
                "Location": location,
                "Age": f"{age} years",
                "Garage": f"{garage} spaces",
                "Pool": "Yes" if pool else "No",
                "Basement": "Yes" if basement else "No",
            }

            for key, value in summary_data.items():
                st.text(f"{key}: {value}")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.header("Feature Importance")

    try:
        feature_importance = pd.DataFrame({
            'Feature': input_df.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)

        fig = go.Figure(data=[
            go.Bar(y=feature_importance['Feature'], x=feature_importance['Importance'], orientation='h')
        ])
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Failed to render feature importance: {e}")

    with st.sidebar:
        st.header("About")
        st.markdown("""
        ### Interactive Property Price Predictor

        Uses machine learning to estimate property prices.

        **Features:**
        - XGBoost regression model
        - Real-time predictions
        - Feature importance analysis
        - Handles categorical & numeric features

        **Model:**
        - Algorithm: XGBoost Regressor
        - Training samples: 400+
        - Features: 8
        """)

        if st.button("Retrain Model"):
            st.info("Retraining model...")
            try:
                generate_sample_properties(DEFAULT_INPUT_FILE, n_samples=500)
                train_model()
                st.success("Model retrained successfully!")
                st.rerun()
            except Exception as e:
                logger.error(f"Retraining failed: {e}")
                st.error(f"Retraining failed: {e}")


if __name__ == "__main__":
    main()
