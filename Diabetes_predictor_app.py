import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. Page Styling with Custom CSS ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

        .header-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2rem;
            font-weight: bold;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header-subtitle {
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            color: #c40233;
        }
        .divider {
            border-top: 1px solid #170225;
            margin: 15px 0;
        }
        .metric-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1rem;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 3px;
        }
        .metric-desc {
            font-family: 'Poppins', sans-serif;
            font-size: 0.8rem;
            color: #bbbbbb;
            margin-bottom: 2px;
        }
        .metric-range {
            font-family: 'Poppins', sans-serif;
            font-size: 0.8rem;
            color: #4CAF50; /* Green */
            margin-bottom: 5px;
        }
        body {
            background-color: #0E1117; /* Dark background */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 2. App Header ---
st.markdown(
    """
    <div class="header-title">
        Diabetes Predictor
    </div>
    <div class="header-subtitle">
        Smart detection of potential diabetes risk.
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True
)

# --- 3. Load the Trained Model ---
try:
    with open("diabetes_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Optional: remove attributes if your model has issues with unpickling certain attributes
    if hasattr(model, 'monotonic_cst'):
        del model.monotonic_cst
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 4. Load the Robust Scaler ---
try:
    with open("robust_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading scaler: {e}")
    st.stop()

# --- 5. Define Feature Inputs ---
# Adjust descriptions and ranges to match your data as needed
feature_info = {
    "Pregnancies": {
        "desc": "Number of times pregnant.",
        "range": "(Typically 0 - 17)"
    },
    "Glucose": {
        "desc": "Plasma glucose concentration (mg/dL).",
        "range": "(80 - 200)"
    },
    "BloodPressure": {
        "desc": "Diastolic blood pressure (mm Hg).",
        "range": "(40 - 120)"
    },
    "SkinThickness": {
        "desc": "Triceps skin fold thickness (mm).",
        "range": "(10 - 99)"
    },
    "Insulin": {
        "desc": "2-Hour serum insulin (μU/mL).",
        "range": "(15 - 276)"
    },
    "BMI": {
        "desc": "Body Mass Index (kg/m²).",
        "range": "(15 - 67)"
    },
    "DiabetesPedigreeFunction": {
        "desc": "Diabetes pedigree function.",
        "range": "(0.078 - 2.42)"
    },
    "Age": {
        "desc": "Age in years.",
        "range": "(21 - 81)"
    }
}

# Collect user inputs in a list
inputs = []

for feature, details in feature_info.items():
    st.markdown(f"<div class='metric-title'>{feature}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-desc'>{details['desc']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-range'>{details['range']}</div>", unsafe_allow_html=True)
    value = st.number_input("", min_value=0.0, format="%.2f", key=feature)
    inputs.append(value)

# Convert inputs to a NumPy array
input_array = np.array(inputs).reshape(1, -1)

# --- 6. Validate and Scale the Inputs ---
if np.any(np.isnan(input_array)):
    st.warning("Please enter valid values for all fields.")
else:
    try:
        input_scaled = scaler.transform(input_array)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # --- 7. Prediction ---
    if st.button("Predict Diabetes"):
        try:
            prediction = model.predict(input_scaled)
            if prediction[0] == 1:
                st.markdown("🚨 **High Risk of Diabetes**<br/>_Please consult with a healthcare provider._",
                            unsafe_allow_html=True)
            else:
                st.markdown("🩺 **Low Risk of Diabetes**<br/>_Your results are within normal range._",
                            unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
