import streamlit as st
import pandas as pd
import joblib        # ← NEW: use joblib instead of pickle
import os
from sklearn.preprocessing import LabelEncoder

# ---------- MODEL PATH ----------
# MODEL_PATH = r"C:\Users\HARSH\Desktop\Streamlit\Fraud Analysis\final_mod.joblib"  # ← .joblib

# # ---------- LOAD MODEL ----------
# @st.cache_resource(show_spinner=False)
# def load_model(path: str):
#     if not os.path.exists(path):
#         st.error(f"❌ Model file not found: {path}")
#         st.stop()
#     return joblib.load(path)        # ← NEW: joblib.load

# Load model

model = joblib.load("final_model.joblib")

# ---------- Streamlit UI ----------
st.title("🚦 Insurance Fraud Detection App")
st.write(
    "Fill in the claim details below and click **Predict** to see whether "
    "the model flags the claim as fraudulent."
)

# Input widgets
fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
base_policy = st.selectbox("Base Policy", ["Liability", "Collision", "All Perils"])
address_change = st.selectbox(
    "Address Change-Claim",
    ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years", "over 8 years"],
)
policy_type = st.selectbox(
    "Policy Type",
    [
        "Sedan - Liability", "Sedan - Collision", "Sedan - All Perils",
        "Sport - Liability", "Sport - Collision", "Sport - All Perils",
        "Utility - Liability", "Utility - Collision", "Utility - All Perils",
    ],
)
year = st.number_input("Policy Year", min_value=1900, max_value=2100, value=2025, step=1)
age_holder = st.selectbox(
    "Age Of Policy Holder",
    ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
     "36 to 40", "41 to 50", "51 to 65", "over 65"],
)
deductible = st.number_input("Deductible ($)", min_value=0, max_value=10000, value=500, step=100)
month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
accident_area = st.selectbox("Accident Area", ["Urban", "Rural"])
past_claims = st.selectbox("Past Number Of Claims", ["none", "1", "2 to 4", "more than 4"])

# ---------- PREDICT ----------
if st.button("Predict"):
    try:
        input_dict = {
            "Fault": fault,
            "BasePolicy": base_policy,
            "AddressChange-Claim": address_change,
            "PolicyType": policy_type,
            "Year": year,
            "AgeOfPolicyHolder": age_holder,
            "Deductible": deductible,
            "Month": month,
            "AccidentArea": accident_area,
            "PastNumberOfClaims": past_claims,
        }
        input_df = pd.DataFrame([input_dict])

        # If you saved ONLY the model (no preprocessing pipeline), keep this encoding block.
        # If your .joblib file is a full preprocessing+model pipeline, comment out or remove it.
        for col in input_df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col])

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Display result
        if prediction == 1:
            st.error("⚠️ The claim is predicted **FRAUDULENT**")
        else:
            st.success("✅ The claim is predicted **GENUINE**")

        # Show details
        with st.expander("🔍 See input data"):
            st.write(input_df)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

