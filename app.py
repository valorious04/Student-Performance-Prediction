import os
import json
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    PROJECT_DIR, "artifacts", "student_predictions.db"
)

def initialize_database():
    """Create the prediction history table if it does not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                predicted_g3 REAL NOT NULL,
                student_data TEXT NOT NULL
            )
        """)

def save_prediction(input_data, prediction):
    """Save one prediction and its input values permanently."""
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO predictions
            (prediction_time, predicted_g3, student_data)
            VALUES (?, ?, ?)
            """,
            (
                prediction_time,
                round(float(prediction), 2),
                json.dumps(input_data)
            )
        )

    return prediction_time

def get_prediction_history():
    """Retrieve saved predictions as a DataFrame."""
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT prediction_time, predicted_g3, student_data
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

    records = []

    for prediction_time, predicted_g3, student_data in rows:
        record = json.loads(student_data)
        record["Prediction_Time"] = prediction_time
        record["Predicted_G3"] = predicted_g3
        records.append(record)

    return pd.DataFrame(records)

def clear_prediction_history():
    """Delete all saved prediction records."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM predictions")


initialize_database()

MODEL_PATH = os.path.join(
    PROJECT_DIR, "artifacts", "day07_optimized_pipeline.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the optimized model: {e}")
    st.stop()

# ---------------- HEADER ----------------
st.title("🎓 Student Performance Prediction System")
st.markdown(
    "### Beginning-of-Term Academic Performance Prediction"
)
st.caption(
    "Predicting final mathematics grades using student "
    "academic, family and personal characteristics."
)
st.divider()


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Home", "Predict Performance", "Prediction History", "About"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.subheader("Welcome!")
    st.write(
        "This application estimates a student's final "
        "mathematics grade using an optimized Random Forest model."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Input Features", "29")
    col2.metric("Model", "Random Forest")
    col3.metric("Dataset Records", "395")

    st.info(
        "Select 'Predict Performance' from the sidebar "
        "to enter student information."
    )

# ---------------- PREDICTION FORM ----------------
elif page == "Predict Performance":

    st.subheader("Student Information")
    st.write("Enter the student's details in all four sections.")

    with st.form("student_prediction_form"):

        # 1. STUDENT INFORMATION
        st.markdown("### 👤 Student Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            school = st.selectbox("School", ["GP", "MS"])
            sex = st.selectbox("Gender", ["F", "M"])
            age = st.selectbox("Age", list(range(15, 23)))

        with c2:
            address = st.selectbox("Address", ["U", "R"])
            famsize = st.selectbox("Family Size", ["GT3", "LE3"])
            Pstatus = st.selectbox("Parents' Cohabitation Status", ["T", "A"])

        with c3:
            Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4])
            Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4])

        st.divider()

        # 2. FAMILY BACKGROUND
        st.markdown("### 👨‍👩‍👧 Family Background")
        c1, c2, c3 = st.columns(3)

        with c1:
            Mjob = st.selectbox(
                "Mother's Job",
                ["at_home", "health", "other", "services", "teacher"]
            )
            Fjob = st.selectbox(
                "Father's Job",
                ["at_home", "health", "other", "services", "teacher"]
            )

        with c2:
            reason = st.selectbox(
                "Reason for Choosing School",
                ["course", "home", "other", "reputation"]
            )
            guardian = st.selectbox(
                "Guardian",
                ["father", "mother", "other"]
            )

        with c3:
            famsup = st.selectbox("Family Educational Support", ["yes", "no"])
            schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])

        st.divider()

        # 3. ACADEMIC INFORMATION
        st.markdown("### 📚 Academic Information")
        c1, c2, c3 = st.columns(3)

        with c1:
            traveltime = st.selectbox(
                "Travel Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "<15 min", 2: "15–30 min",
                    3: "30–60 min", 4: ">60 min"
                }[x]
            )
            studytime = st.selectbox(
                "Weekly Study Time",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "<2 hours", 2: "2–5 hours",
                    3: "5–10 hours", 4: ">10 hours"
                }[x]
            )

        with c2:
            failures = st.selectbox(
                "Past Class Failures",
                [0, 1, 2, 3]
            )
            paid = st.selectbox("Extra Paid Classes", ["yes", "no"])

        with c3:
            higher = st.selectbox(
                "Wants Higher Education", ["yes", "no"]
            )
            activities = st.selectbox(
                "Extracurricular Activities", ["yes", "no"]
            )

        st.divider()

        # 4. LIFESTYLE AND WELL-BEING
        st.markdown("### ❤️ Lifestyle and Well-being")
        c1, c2, c3 = st.columns(3)

        with c1:
            nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
            internet = st.selectbox("Internet Access at Home", ["yes", "no"])
            romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])

        with c2:
            famrel = st.select_slider(
                "Family Relationship Quality", options=[1, 2, 3, 4, 5]
            )
            freetime = st.select_slider(
                "Free Time After School", options=[1, 2, 3, 4, 5]
            )
            goout = st.select_slider(
                "Going Out with Friends", options=[1, 2, 3, 4, 5]
            )

        with c3:
            Dalc = st.select_slider(
                "Workday Alcohol Consumption", options=[1, 2, 3, 4, 5]
            )
            Walc = st.select_slider(
                "Weekend Alcohol Consumption", options=[1, 2, 3, 4, 5]
            )
            health = st.select_slider(
                "Current Health Status", options=[1, 2, 3, 4, 5]
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Student Performance",
            use_container_width=True
        )

    if submitted:
        input_data = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health
        }

        input_df = pd.DataFrame([input_data])

        try:
            prediction = float(model.predict(input_df)[0])

            # Save this successful prediction permanently in SQLite.
            save_prediction(input_data, prediction)

            st.success("Prediction generated successfully!")

            st.markdown("### 📊 Prediction Result")

            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(
                    "Predicted Final Grade",
                    f"{prediction:.2f} / 20"
                    )

            with col2:
                st.progress(
                    max(0.0, min(prediction / 20, 1.0))
                )
                st.caption("Predicted grade on a 20-point scale")

            st.info(
                "This prediction is an estimate based on historical "
                "student data. It should not be treated as an official "
                "grade or a guarantee of academic performance."
            )

            st.warning(
                "The model has moderate predictive performance. "
                "Use its output as an indicative estimate, not as the "
                "sole basis for academic decisions."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------- PREDICTION HISTORY ----------------
elif page == "Prediction History":
    st.subheader("📋 Prediction History")

    history_df = get_prediction_history()

    history = not history_df.empty

    if history:

        st.metric("Total Predictions", len(history_df))

        # Put timestamp and predicted grade first for easier review.
        preferred_columns = ["Prediction_Time", "Predicted_G3"]
        remaining_columns = [
            column for column in history_df.columns
            if column not in preferred_columns
        ]
        history_df = history_df[preferred_columns + remaining_columns]

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Prediction History (CSV)",
            data=csv_data,
            file_name="student_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🗑️ Clear Prediction History"):
            clear_prediction_history()
            st.success("Prediction history cleared successfully.")
            st.rerun()
    else:
        st.info(
            "No predictions available yet. "
            "Generate a prediction to populate this history."
        )

# ---------------- ABOUT ----------------
elif page == "About":
    st.subheader("About the Project")
    st.write(
        "A beginning-of-term student performance prediction "
        "system developed using the UCI Student Performance dataset."
    )
    st.write("Model: Optimized Random Forest Regressor")
    st.write("Input Features: 29")
    st.write("Target: Final Mathematics Grade (G3)")
    st.write("Dataset: 395 student records")
