import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEFAULT_MODEL_FILE,
    DEFAULT_SCALER_FILE,
    DEFAULT_ENCODER_FILE,
    DEFAULT_INPUT_FILE,
    MODEL_DIR,
)
from src.sample_data import generate_sample_properties
from src.preprocessing import preprocess
from src.train import main as train_model


st.set_page_config(page_title="Property Price Predictor", layout="wide", initial_sidebar_state="expanded")


def load_models():
    """Load trained model and preprocessors."""
    if not DEFAULT_MODEL_FILE.exists():
        st.warning("Model not found. Training model on demo data...")
        if not DEFAULT_INPUT_FILE.exists():
            generate_sample_properties(DEFAULT_INPUT_FILE)
        train_model()

    model = load(DEFAULT_MODEL_FILE)
    scaler = load(DEFAULT_SCALER_FILE)
    encoder = load(DEFAULT_ENCODER_FILE)
    return model, scaler, encoder


def create_prediction_df(sqft, bedrooms, bathrooms, location, age, garage, pool, basement, scaler, encoder):
    """Create a DataFrame for prediction."""
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

    numeric_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'garage', 'pool', 'basement']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    location_df = pd.DataFrame({'location': [location]})
    encoded = encoder.transform(location_df)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['location']))

    input_data = input_data.drop(columns=['location'])
    input_data = pd.concat([input_data, encoded_df], axis=1)

    return input_data


def main():
    st.title("🏡 Interactive Property Price Predictor")
    st.markdown("Predict residential property prices using machine learning (XGBoost)")

    model, scaler, encoder = load_models()

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Property Details")
        sqft = st.slider("Square Footage", 500, 10000, 2500, 100)
        bedrooms = st.slider("Bedrooms", 1, 10, 4, 1)
        bathrooms = st.slider("Bathrooms", 1.0, 5.0, 2.5, 0.5)
        location = st.selectbox("Location", ["Downtown", "Suburbs", "Waterfront", "Historic", "Hillside"])
        age = st.slider("Age (Years)", 0, 100, 20, 1)
        garage = st.slider("Garage Spaces", 0, 4, 2, 1)
        pool = st.checkbox("Has Pool", value=False)
        basement = st.checkbox("Has Basement", value=False)

    with col2:
        st.header("Price Prediction")

        input_df = create_prediction_df(
            sqft, bedrooms, bathrooms, location, age, garage,
            int(pool), int(basement), scaler, encoder
        )

        predicted_price = model.predict(input_df)[0]

        col_price, col_metric = st.columns([2, 1])
        with col_price:
            st.metric(
                "Estimated Price",
                f"${predicted_price:,.0f}",
                delta=None,
            )

        st.markdown("---")
        st.subheader("Property Summary")
        summary_data = {
            "Square Footage": f"{sqft:,} sqft",
            "Bedrooms": f"{bedrooms}",
            "Bathrooms": f"{bathrooms}",
            "Location": location,
            "Age": f"{age} years",
            "Garage": f"{garage} spaces",
            "Pool": "✓" if pool else "✗",
            "Basement": "✓" if basement else "✗",
        }

        for key, value in summary_data.items():
            st.text(f"{key}: {value}")

    st.markdown("---")
    st.header("Feature Importance")

    feature_importance = pd.DataFrame({
        'Feature': input_df.columns,
        'Importance': model.feature_importances_[:len(input_df.columns)]
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

        if st.button("🔄 Retrain Model"):
            st.info("Retraining model...")
            generate_sample_properties(DEFAULT_INPUT_FILE, n_samples=500)
            train_model()
            st.success("Model retrained successfully!")
            st.rerun()


if __name__ == "__main__":
    main()
