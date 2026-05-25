import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Classifier", layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "svm_classification_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
DATA_PATH = os.path.join(BASE_DIR, "student_performance_dataset.csv")


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    if not os.path.exists(METADATA_PATH):
        st.error(f"Metadata file '{METADATA_PATH}' not found.")
        st.stop()

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, metadata = load_artifacts()
feature_columns = metadata["feature_columns"]
class_names = metadata["class_names"]

st.title("Student Performance Classifier")
st.write("Predict performance category (Low / Medium / High) using a support vector machine.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 30, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_type = st.selectbox("City Type", ["Urban", "Semi-Urban", "Rural"])
    study_hours = st.slider("Study Hours Per Day", 0.0, 10.0, 5.0, 0.1)
    sleep_hours = st.slider("Sleep Hours Per Night", 3.0, 12.0, 7.0, 0.1)

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 6)
    focus_score = st.slider("Focus Score (1-10)", 1, 10, 6)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    assignment_completion = st.slider("Assignment Completion (%)", 0, 100, 80)

input_df = pd.DataFrame(
    {
        "age": [age],
        "gender": [gender],
        "city_type": [city_type],
        "study_hours_per_day": [study_hours],
        "sleep_hours": [sleep_hours],
        "stress_level": [stress_level],
        "motivation_level": [motivation_level],
        "focus_score": [focus_score],
        "attendance_percentage": [attendance],
        "assignment_completion_rate": [assignment_completion],
    }
)

input_processed = pd.get_dummies(input_df, drop_first=True)
for col in feature_columns:
    if col not in input_processed.columns:
        input_processed[col] = 0
input_processed = input_processed[feature_columns]

if st.button("Predict Performance Category", use_container_width=True):
    prediction = model.predict(input_processed)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_processed)[0]
        prob_df = pd.DataFrame(
            {
                "Class": class_names,
                "Probability": [float(p) for p in probs],
            }
        ).sort_values("Probability", ascending=False)
    else:
        prob_df = pd.DataFrame({"Class": class_names, "Probability": [0.0] * len(class_names)})

    st.subheader("Prediction")
    st.success(f"Predicted category: {prediction}")

    st.subheader("Class probabilities")
    st.dataframe(prob_df, use_container_width=True)

st.sidebar.header("Model info")
st.sidebar.write(f"Validation accuracy: {metadata.get('accuracy', 'N/A')}")
st.sidebar.write(f"Macro F1 score: {metadata.get('f1_macro', 'N/A')}")
st.sidebar.write("Target classes: Low, Medium, High")

if st.sidebar.checkbox("Show raw dataset preview"):
    st.dataframe(load_data().head(20), use_container_width=True)