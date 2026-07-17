from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="CHURN",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent


def candidate_paths(filename):
    return [
        APP_DIR / "models" / filename,
        TASK_DIR / "models" / filename,
        TASK_DIR / "code file" / "models" / filename,
        APP_DIR / filename,
    ]


def find_dataset():
    dataset_dir = TASK_DIR / "datasets"
    preferred = dataset_dir / "Churn_Modelling.csv"
    if preferred.exists():
        return preferred
    csv_files = sorted(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
    return csv_files[0] if csv_files else None


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@st.cache_resource
def load_or_train_model():
    model_filename = "customer_churn_best_model.joblib"
    for path in candidate_paths(model_filename):
        if path.exists():
            return joblib.load(path), f"Loaded saved model from {path}"

    dataset_path = find_dataset()
    if dataset_path is None:
        return None, "No saved model or dataset was found."

    df = pd.read_csv(dataset_path)
    if "Exited" not in df.columns:
        return None, "Dataset found, but target column 'Exited' was missing."

    drop_cols = [col for col in ["RowNumber", "CustomerId", "Surname"] if col in df.columns]
    X = df.drop(columns=drop_cols + ["Exited"])
    y = df["Exited"].astype(int)

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    model.fit(X_train, y_train)
    return model, f"Trained a model from {dataset_path}"


model, model_status = load_or_train_model()


st.title("Customer Churn Prediction")
st.caption("Predict whether a bank customer is likely to leave the service.")

with st.sidebar:
    st.header("Project")
    st.write("Codsoft ML Internship")
    st.write("Model: Scikit-learn pipeline")
    st.info(model_status)


if model is None:
    st.error(
        "Model is not available. Run the notebook first to create "
        "`customer_churn_best_model.joblib`, or place `Churn_Modelling.csv` in the datasets folder."
    )
    st.stop()


left, right = st.columns([1, 1])

with left:
    st.subheader("Customer Details")
    with st.form("churn_form"):
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 95, 35)
        tenure = st.slider("Tenure", 0, 10, 5)
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=75000.0, step=1000.0)
        submitted = st.form_submit_button("Predict Churn")

customer = pd.DataFrame(
    [
        {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": estimated_salary,
        }
    ]
)

with right:
    st.subheader("Prediction")
    if submitted:
        if hasattr(model, "predict_proba"):
            churn_probability = float(model.predict_proba(customer)[0, 1])
        else:
            churn_probability = float(model.predict(customer)[0])

        prediction = "Likely to Churn" if churn_probability >= 0.50 else "Likely to Stay"
        risk_level = "High" if churn_probability >= 0.70 else "Medium" if churn_probability >= 0.40 else "Low"

        st.metric("Churn Probability", f"{churn_probability * 100:.2f}%")
        st.metric("Risk Level", risk_level)

        if churn_probability >= 0.50:
            st.warning(prediction)
        else:
            st.success(prediction)

        chart_df = pd.DataFrame(
            {"Outcome": ["Stay", "Churn"], "Probability": [1 - churn_probability, churn_probability]}
        )
        st.bar_chart(chart_df, x="Outcome", y="Probability")
    else:
        st.write("Enter customer details and click **Predict Churn**.")


st.subheader("Input Preview")
st.dataframe(customer, use_container_width=True)
