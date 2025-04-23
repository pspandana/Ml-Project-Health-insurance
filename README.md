# Health Insurance Premium Predictor

A machine learning application that predicts insurance premiums based on various factors including age, BMI, smoking status, and more.

## Features

- Age-specific models for better prediction accuracy
- Support for both young (≤25) and older customers
- Risk score calculation based on multiple factors
- Interactive web interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pspandana/Ml-Project-Health-insurance.git
cd Ml-Project-Health-insurance
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit web application
- `premium_predictor.py`: Core prediction logic and model handling
- `artifacts/`: Directory containing trained models and scalers
  - `model_young.joblib`: Model for customers ≤25 years
  - `model_rest.joblib`: Model for customers >25 years
  - `scaler_young.joblib`: Feature scaler for young customer model
  - `scaler_rest.joblib`: Feature scaler for older customer model

## Models

The application uses two separate XGBoost models:
1. Young customer model (age ≤ 25)
2. Regular customer model (age > 25) with additional genetic risk factor
