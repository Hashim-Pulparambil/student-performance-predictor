import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

st.title("🎓 Student Performance Predictor")

st.write("App started successfully!")

try:
    model = joblib.load("models/best_model.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.exception(e)
    st.stop()

gender = st.selectbox("Gender", ["female", "male"])

race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

education = st.selectbox(
    "Parental Level of Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school",
    ],
)

lunch = st.selectbox(
    "Lunch",
    ["standard", "free/reduced"],
)

prep = st.selectbox(
    "Test Preparation",
    ["completed", "none"],
)

if st.button("Predict"):
    sample = pd.DataFrame(
        {
            "gender": [gender],
            "race/ethnicity": [race],
            "parental level of education": [education],
            "lunch": [lunch],
            "test preparation course": [prep],
        }
    )

    prediction = model.predict(sample)[0]

    st.success(f"Prediction: {prediction}")