import streamlit as st
from premium_predictor import PremiumPredictor
from typing import Dict, Any

# Initialize the predictor
try:
    predictor = PremiumPredictor()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error initializing predictor: {str(e)}")
    st.stop()

# Set page title
st.set_page_config(page_title="Insurance Premium Predictor", layout="wide")

# Add a title and description
st.title("Insurance Premium Prediction")
st.write("Enter your details below to get an estimated insurance premium.")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=35, value=25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Northwest", "Northeast", "Southwest", "Southeast"])
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obesity", "Underweight"])
        
    with col2:
        smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Regular", "Occasional"])
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])
        income_lakhs = st.number_input("Annual Income (in lakhs)", min_value=0.0, max_value=100.0, value=5.0)
        medical_history = st.selectbox("Medical History", 
                                     ["No Disease", "Diabetes", "Heart Disease", "High Blood Pressure", "Thyroid"])
        insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])
        
        genetical_risk = 0.0
        if age > 25:
            genetical_risk = st.number_input("Genetical Risk Score (0-1)", min_value=0.0, max_value=1.0, value=0.0, help="This score represents the genetic risk factor based on family history and genetic testing.")

    submitted = st.form_submit_button("Predict Premium")

if submitted:
    try:
        # Prepare input data
        input_data: Dict[str, Any] = {
            # Numeric features
            'age': age,
            'number_of_dependants': num_dependents,
            'income_lakhs': income_lakhs,
            'insurance_plan': insurance_plan,
            
            # One-hot encoded features
            'gender_Male': 1 if gender == "Male" else 0,
            'region_Northwest': 1 if region == "Northwest" else 0,
            'region_Southeast': 1 if region == "Southeast" else 0,
            'region_Southwest': 1 if region == "Southwest" else 0,
            'marital_status_Unmarried': 1 if marital_status == "Unmarried" else 0,
            'bmi_category_Obesity': 1 if bmi_category == "Obesity" else 0,
            'bmi_category_Overweight': 1 if bmi_category == "Overweight" else 0,
            'bmi_category_Underweight': 1 if bmi_category == "Underweight" else 0,
            'smoking_status_Occasional': 1 if smoking_status == "Occasional" else 0,
            'smoking_status_Regular': 1 if smoking_status == "Regular" else 0,
            'employment_status_Salaried': 1 if employment_status == "Salaried" else 0,
            'employment_status_Self-Employed': 1 if employment_status == "Self-Employed" else 0,
            
            # Additional features
            'genetical_risk': genetical_risk if age > 25 else 0.0
        }
        
        # Get prediction
        prediction = predictor.predict(input_data)
        
        # Display prediction
        st.success(f"Estimated Annual Premium: ₹{prediction:,.2f}")
        
        # Add some context about the prediction
        st.info("""
        Factors that typically increase premium:
        - Higher age
        - Smoking status
        - Medical conditions
        - Higher BMI
        - Insurance plan type
        """)
        
    except ValueError as e:
        st.error(f"Error with input data: {str(e)}")
        st.error("Debug info - Input data:")
        st.write(input_data)
    except Exception as e:
        st.error(f"Unexpected error during prediction: {str(e)}")
        st.error("Please contact support with this error message.")
