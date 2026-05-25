# Student Performance Prediction - SVM Regression Model

A machine learning project that predicts a student's final exam score using a support vector machine regression model.

## Project overview

This project mirrors the linear regression app structure, but uses an SVM regression model. It includes:
- A Jupyter notebook for training and evaluation
- A Streamlit app for interactive predictions
- Model and metadata artifacts for deployment

## Dataset

The project uses `student_performance_dataset.csv` with the target column:
- `final_exam_score`

The model is trained with these input features:
- age
- gender
- city_type
- study_hours_per_day
- sleep_hours
- stress_level
- motivation_level
- focus_score
- attendance_percentage
- assignment_completion_rate

## Project structure

regression/
- app.py
- requirements.txt
- readme.md
- student_performance_dataset.csv
- svm_regression_model.pkl
- model_metadata.json

## How to run

1. Install dependencies:

pip install -r requirements.txt

2. Train and export model if needed:

Open the regression notebook and run all cells.

3. Run the Streamlit app:

streamlit run app.py

## Notes

- The app applies the same one-hot encoding pattern used in training.
- Feature alignment is handled with the saved metadata.