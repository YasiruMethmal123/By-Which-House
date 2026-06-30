import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/feature.pkl")

# =========================
# HEADER
# =========================

st.title("🏠 House Price Category Prediction")

st.markdown(
    """
Predict whether a property belongs to the **High Price** or **Low Price** category
using the trained Machine Learning model.
"""
)

st.divider()

# =========================
# SIDEBAR
# =========================

st.sidebar.header("About")

st.sidebar.info("""
Model: Logistic Regression

Features:
- Location
- Size
- Bedrooms

Output:
- High Price
- Low Price
""")

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox(
        " Location",
        [
            "Auckland",
            "Beijing",
            "Berlin",
            "Chicago",
            "Houston",
            "London",
            "Los Angeles",
            "Madrid",
            "Melbourne",
            "New York",
            "Osaka",
            "Paris",
            "Phoenix",
            "Rome",
            "Shanghai",
            "Singapore",
            "Sydney",
            "Tokyo",
            "Toronto",
            "Vancouver"
        ]
    )

    size = st.slider(
        "📐 House Size",
        min_value=500,
        max_value=5000,
        value=1500,
        step=50
    )

with col2:

    bedrooms = st.slider(
        "🛏 Bedrooms",
        min_value=1,
        max_value=10,
        value=3
    )

# =========================
# PREDICTION
# =========================

if st.button(" Predict", use_container_width=True):

    try:

        # Encode location
        encoded_location = encoder.transform([[location]])

        # Create input dataframe
        sample = pd.DataFrame({
            "Size": [size],
            "Bedrooms": [bedrooms]
        })

        # Scale numeric features
        sample_scaled = scaler.transform(sample)

        # Combine encoded location + scaled features
        import numpy as np

        final_input = np.hstack([
            encoded_location,
            sample_scaled
        ])

        prediction = model.predict(final_input)[0]

        probability = model.predict_proba(final_input)[0]

        st.divider()

        # =========================
        # RESULT
        # =========================

        if prediction == 1:
            st.success(" HIGH PRICE HOUSE")
        else:
            st.error(" LOW PRICE HOUSE")

        # =========================
        # CHARTS
        # =========================

        left, right = st.columns(2)

        with left:

            confidence = float(max(probability) * 100)

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    title={"text": "Confidence"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "#ffcccc"},
                            {"range": [50, 80], "color": "#fff3cd"},
                            {"range": [80, 100], "color": "#d4edda"}
                        ]
                    }
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        with right:

            prob_df = pd.DataFrame({
                "Category": ["Low Price", "High Price"],
                "Probability": [
                    probability[0],
                    probability[1]
                ]
            })

            pie = px.pie(
                prob_df,
                values="Probability",
                names="Category",
                hole=0.55,
                title="Prediction Probability"
            )

            st.plotly_chart(
                pie,
                use_container_width=True
            )

        # =========================
        # SUMMARY
        # =========================

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Location",
                "Size",
                "Bedrooms"
            ],
            "Value": [
                location,
                size,
                bedrooms
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.subheader(" Probability Scores")

        st.write(
            f"High Price: **{probability[1]*100:.2f}%**"
        )

        st.write(
            f"Low Price: **{probability[0]*100:.2f}%**"
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")