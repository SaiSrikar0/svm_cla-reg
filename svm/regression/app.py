import json
import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="📚", layout="wide")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "svm_regression_model.pkl")
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
def load_training_data():
    return pd.read_csv(DATA_PATH)


model, metadata = load_artifacts()
feature_columns = metadata["feature_columns"]

st.title("Student Performance Predictor")
st.write("Predict final exam score using a support vector machine regression model.")

st.divider()

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

input_data = pd.DataFrame(
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

try:
    input_processed = pd.get_dummies(input_data, drop_first=True)

    training_data = load_training_data()
    if training_data is not None:
        training_data_processed = training_data.drop("final_exam_score", axis=1)
        training_data_processed = pd.get_dummies(training_data_processed, drop_first=True)

        for col in training_data_processed.columns:
            if col not in input_processed.columns:
                input_processed[col] = 0

        input_processed = input_processed[feature_columns]
except Exception as e:
    st.error(f"Data preprocessing error: {str(e)}")

if st.button("Predict Final Exam Score", use_container_width=True):
    try:
        prediction = model.predict(input_processed)
        pred_score = prediction[0]
        pred_score = max(0, min(100, pred_score))

        st.divider()
        st.subheader("Prediction Result")

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Predicted Final Exam Score", f"{pred_score:.1f}/100")

        with col_metric2:
            if pred_score >= 85:
                performance = "Excellent"
            elif pred_score >= 70:
                performance = "Good"
            elif pred_score >= 60:
                performance = "Average"
            else:
                performance = "Needs Improvement"

            st.metric("Performance Level", performance)

        st.progress(pred_score / 100)

        if pred_score >= 85:
            st.balloons()
            st.success("Excellent! Keep up the great work!")
        elif pred_score >= 70:
            st.info("Good performance! Focus on consistency.")
        elif pred_score >= 60:
            st.warning("Average performance. Consider improving study habits.")
        else:
            st.error("Below average. Seek additional support or tutoring.")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info(f"Debug info - Input shape: {input_processed.shape if 'input_processed' in locals() else 'N/A'}")

st.sidebar.header("About This App")

st.sidebar.markdown(
    """
### How It Works:
1. Input student information such as study hours, sleep, stress, and attendance.
2. Features are encoded to match the model's training format.
3. The SVM regression model predicts the final exam score.
4. Results are mapped to simple performance levels.

### Model Performance:
- Trained on student performance data
- Features: behavioral and academic variables
- Target: Final exam score (0-100)

### Tips for Better Predictions:
- Provide realistic study and lifestyle values
- Keep categorical inputs consistent with training data
- Attendance and assignment completion are strong predictors

**Technology**: Streamlit + Scikit-learn
"""
)