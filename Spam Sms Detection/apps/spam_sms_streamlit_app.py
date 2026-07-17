import re
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


st.set_page_config(
    page_title="Spam SMS Detection",
    page_icon="SMS",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def candidate_paths(filename):
    return [
        APP_DIR / "models" / filename,
        TASK_DIR / "models" / filename,
        TASK_DIR / "code file" / "models" / filename,
        APP_DIR / filename,
    ]


def find_dataset():
    dataset_dir = TASK_DIR / "datasets"
    preferred = dataset_dir / "spam.csv"
    if preferred.exists():
        return preferred
    csv_files = sorted(dataset_dir.glob("*.csv")) if dataset_dir.exists() else []
    return csv_files[0] if csv_files else None


@st.cache_resource
def load_or_train_model():
    model_filename = "spam_sms_classifier.joblib"
    for path in candidate_paths(model_filename):
        if path.exists():
            return joblib.load(path), f"Loaded saved model from {path}", None

    dataset_path = find_dataset()
    if dataset_path is None:
        return None, "No saved model or dataset was found.", None

    df_raw = pd.read_csv(dataset_path, encoding="latin-1")
    if {"v1", "v2"}.issubset(df_raw.columns):
        df = df_raw[["v1", "v2"]].copy()
        df.columns = ["label", "message"]
    else:
        return None, "Dataset found, but expected columns `v1` and `v2` were missing.", None

    df = df.dropna(subset=["label", "message"]).drop_duplicates().reset_index(drop=True)
    df["clean_message"] = df["message"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["target"].astype(int),
        test_size=0.20,
        random_state=42,
        stratify=df["target"].astype(int),
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
    }

    best_model = None
    best_name = None
    best_f1 = -1.0
    best_accuracy = 0.0

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        max_df=0.95,
                        max_features=20_000,
                        sublinear_tf=True,
                    ),
                ),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        score = f1_score(y_test, y_pred, zero_division=0)
        if score > best_f1:
            best_model = pipeline
            best_name = name
            best_f1 = score
            best_accuracy = accuracy_score(y_test, y_pred)

    metrics = {"model": best_name, "f1": best_f1, "accuracy": best_accuracy}
    return best_model, f"Trained a model from {dataset_path}", metrics


def spam_score(model, messages):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(messages)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(messages)
        return 1 / (1 + np.exp(-raw_scores))
    return model.predict(messages)


model, model_status, metrics = load_or_train_model()


st.title("Spam SMS Detection")
st.caption("Classify SMS messages as spam or legitimate.")

with st.sidebar:
    st.header("Project")
    st.write("Codsoft ML Internship")
    st.write("Model: TF-IDF + Scikit-learn classifier")
    st.info(model_status)
    if metrics:
        st.metric("Validation F1", f"{metrics['f1']:.3f}")
        st.metric("Validation Accuracy", f"{metrics['accuracy']:.3f}")


if model is None:
    st.error(
        "Model is not available. Run the notebook first to create "
        "`spam_sms_classifier.joblib`, or place `spam.csv` in the datasets folder."
    )
    st.stop()


left, right = st.columns([1.2, 1])

with left:
    st.subheader("Message")
    message = st.text_area(
        "Enter SMS text",
        value="Congratulations! You have won a free ticket. Call now to claim your prize.",
        height=180,
    )
    predict_button = st.button("Check Message", type="primary")

with right:
    st.subheader("Prediction")
    if predict_button:
        cleaned = clean_text(message)
        prediction = int(model.predict([cleaned])[0])
        score = float(spam_score(model, [cleaned])[0])
        label = "Spam" if prediction == 1 else "Ham"

        st.metric("Spam Score", f"{score * 100:.2f}%")
        if prediction == 1:
            st.error(f"Prediction: {label}")
        else:
            st.success(f"Prediction: {label}")

        chart_df = pd.DataFrame(
            {"Class": ["Ham", "Spam"], "Score": [1 - score, score]}
        )
        st.bar_chart(chart_df, x="Class", y="Score")
    else:
        st.write("Type a message and click **Check Message**.")


st.subheader("Cleaned Text Preview")
st.code(clean_text(message), language="text")

st.subheader("Try These Examples")
examples = [
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been selected for a cash reward. Reply WIN now.",
    "Please call me when you reach home.",
]
for example in examples:
    if st.button(example):
        cleaned_example = clean_text(example)
        example_prediction = int(model.predict([cleaned_example])[0])
        example_score = float(spam_score(model, [cleaned_example])[0])
        st.write(
            {
                "message": example,
                "prediction": "Spam" if example_prediction == 1 else "Ham",
                "spam_score": round(example_score, 4),
            }
        )
