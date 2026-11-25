# Student Performance Prediction System

A machine learning application that estimates students' final mathematics grades using academic, family, demographic, and lifestyle characteristics.

## Project Overview

This project uses the UCI Student Performance Mathematics dataset to develop a beginning-of-term student performance prediction system.

The application accepts 29 student input features and predicts the final mathematics grade (`G3`) using an optimized Random Forest Regression pipeline.

## Key Features

- Beginning-of-term student performance prediction.
- Data preprocessing using Scikit-learn.
- Optimized Random Forest Regressor.
- Interactive Streamlit web application.
- Prediction history stored permanently in SQLite.
- Prediction history available after browser refresh and application restart.
- CSV export of prediction records.
- Clear prediction history functionality.

## Dataset

- Source: UCI Student Performance Dataset
- File: `student-mat.csv`
- Records: 395
- Original columns: 33
- Prediction target: `G3` (final mathematics grade)

The model uses 29 beginning-of-term features.

The following columns are excluded:
- `G1`: First-period grade.
- `G2`: Second-period grade.
- `G3`: Target variable.
- `absences`: Accumulated absence information.

This feature selection is intended to avoid using information that would not be available at the beginning of the academic term.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Joblib
- Streamlit
- SQLite
- Matplotlib and Seaborn
- Jupyter Notebook

## Model Development

The optimized model uses a Scikit-learn Pipeline containing:
1. A preprocessing component.
2. A Random Forest Regressor.

Hyperparameters were optimized using RandomizedSearchCV with five-fold cross-validation.

### Model Evaluation

| Metric | Result |
|---|---:|
| Cross-validation MAE | 3.1720 |
| Cross-validation MAE standard deviation | 0.2507 |
| Test MAE | 3.2525 |
| Test RMSE | 4.0798 |
| Test R² | 0.1882 |

These are development evaluation results, not a guarantee of future prediction accuracy.

## Streamlit Application

The application provides four main sections:

- Home
- Predict Performance
- Prediction History
- About

The prediction form organizes student information into four sections:
- Student Information
- Family Background
- Academic Information
- Lifestyle and Well-being

## SQLite Prediction History

Successful predictions are stored in:

`artifacts/student_predictions.db`

The SQLite database stores:
- Record ID
- Prediction timestamp
- Predicted final grade
- Student input data in JSON format

History can be retrieved after browser refreshes and application restarts. Users can also download the records as CSV or clear the stored history.

## Project Structure

```text
Student-Performance-Prediction/
│
├── app.py
├── app_day11_backup.py
├── dataset/
├── notebooks/
├── artifacts/
│   ├── day07_optimized_pipeline.pkl
│   └── student_predictions.db
├── models/
├── outputs/
├── src/
├── README.md
├── .gitignore
└── student.zip
