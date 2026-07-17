# Streamlit Apps for Codsoft ML Tasks

This folder contains two optional dashboard apps:

- `customer_churn_streamlit_app.py`
- `spam_sms_streamlit_app.py`

## Where to Put Them

For Customer Churn:

```text
CODSOFT/
  Customer Churn/
    app/
      customer_churn_streamlit_app.py
```

For Spam SMS Detection:

```text
CODSOFT/
  Spam Sms Detection/
    app/
      spam_sms_streamlit_app.py
```

## Dataset Locations

Customer Churn:

```text
CODSOFT/Customer Churn/datasets/Churn_Modelling.csv
```

Spam SMS:

```text
CODSOFT/Spam Sms Detection/datasets/spam.csv
```

## How to Run

Open a terminal in the matching task folder and run:

```bash
pip install -r requirements.txt
streamlit run app/customer_churn_streamlit_app.py
```

or:

```bash
pip install -r requirements.txt
streamlit run app/spam_sms_streamlit_app.py
```

Each app first tries to load the model saved by the notebook. If the saved model is not available, it trains a model from the dataset file.
