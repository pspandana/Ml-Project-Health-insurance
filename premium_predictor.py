import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import xgboost as xgb
from typing import Dict, Any

class PremiumPredictor:
    def __init__(self):
        # Load both models and scalers
        self.young_model = joblib.load('artifacts/model_young.joblib')
        self.rest_model = joblib.load('artifacts/model_rest.joblib')
        
        young_scaler_dict = joblib.load('artifacts/scaler_young.joblib')
        rest_scaler_dict = joblib.load('artifacts/scaler_rest.joblib')
        
        self.young_scaler = young_scaler_dict['scaler']
        self.rest_scaler = rest_scaler_dict['scaler']
        
        # Define mappings
        self.insurance_plan_map = {
            'Bronze': 1,
            'Silver': 2,
            'Gold': 3
        }
        
        # Get feature names from both models
        self.young_feature_names = self.young_model.get_booster().feature_names
        self.rest_feature_names = self.rest_model.get_booster().feature_names
        
        print("Young model features:", self.young_feature_names)
        print("Rest model features:", self.rest_feature_names)

    def calculate_risk_score(self, input_data: Dict[str, Any]) -> float:
        """Calculate normalized risk score based on input features."""
        base_score = 0.0
        
        # Age factor (higher age = higher risk)
        age_factor = (input_data['age'] - 18) / (35 - 18)  # Normalize between 18-35
        base_score += age_factor * 0.2
        
        # BMI category factor
        if input_data['bmi_category_Obesity']:
            base_score += 0.2
        elif input_data['bmi_category_Overweight']:
            base_score += 0.1
        elif input_data['bmi_category_Underweight']:
            base_score += 0.1
        
        # Smoking status factor
        if input_data['smoking_status_Regular']:
            base_score += 0.3
        elif input_data['smoking_status_Occasional']:
            base_score += 0.15
        
        # Dependents factor
        dependents_factor = min(input_data['number_of_dependants'] / 5.0, 1.0)
        base_score += dependents_factor * 0.1
        
        # Normalize final score between 0 and 1
        return min(max(base_score, 0.0), 1.0)

    def prepare_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model prediction."""
        print("\nInput data:", input_data)
        
        # Calculate normalized risk score
        risk_score = self.calculate_risk_score(input_data)
        
        # Get the appropriate feature names based on age
        feature_names = self.rest_feature_names if input_data['age'] > 25 else self.young_feature_names
        
        # Create a DataFrame with all required features
        features = pd.DataFrame(index=[0])
        
        # Add numeric features
        features['age'] = int(input_data['age'])
        features['number_of_dependants'] = int(input_data['number_of_dependants'])
        features['income_level'] = int(input_data['income_lakhs'])  # For scaler compatibility
        features['income_lakhs'] = int(input_data['income_lakhs'])  # For scaler compatibility
        features['insurance_plan'] = int(self.insurance_plan_map[input_data['insurance_plan']])
        features['normalized_risk_score'] = risk_score
        
        # Add binary features
        features['gender_Male'] = int(input_data['gender_Male'])
        features['region_Northwest'] = int(input_data['region_Northwest'])
        features['region_Southeast'] = int(input_data['region_Southeast'])
        features['region_Southwest'] = int(input_data['region_Southwest'])
        features['marital_status_Unmarried'] = int(input_data['marital_status_Unmarried'])
        features['bmi_category_Obesity'] = int(input_data['bmi_category_Obesity'])
        features['bmi_category_Overweight'] = int(input_data['bmi_category_Overweight'])
        features['bmi_category_Underweight'] = int(input_data['bmi_category_Underweight'])
        features['smoking_status_Occasional'] = int(input_data['smoking_status_Occasional'])
        features['smoking_status_Regular'] = int(input_data['smoking_status_Regular'])
        features['employment_status_Salaried'] = int(input_data['employment_status_Salaried'])
        features['employment_status_Self-Employed'] = int(input_data['employment_status_Self-Employed'])
        
        # Add genetical_risk for rest model if age > 25
        if input_data['age'] > 25:
            features['genetical_risk'] = float(input_data.get('genetical_risk', 0.0))
        
        # Scale numeric features first
        numeric_cols = ['age', 'number_of_dependants', 'income_level', 'income_lakhs', 'insurance_plan']
        numeric_features = features[numeric_cols].copy()
        
        # Use appropriate scaler based on age
        scaler = self.young_scaler if input_data['age'] <= 25 else self.rest_scaler
        features[numeric_cols] = scaler.transform(numeric_features)
        
        # Then select features in correct order
        features = features[feature_names]
        
        print("\nFeatures after preparation:")
        print("Shape:", features.shape)
        print("Columns:", features.columns.tolist())
        print("Data types:")
        print(features.dtypes)
        print("\nFeatures contents:")
        print(features)
        
        return features

    def predict(self, input_data: Dict[str, Any]) -> float:
        """Make a prediction for the given input data."""
        try:
            print("\nStarting prediction...")
            # Choose appropriate model based on age
            model = self.young_model if input_data['age'] <= 25 else self.rest_model
            print("Model feature names:", model.get_booster().feature_names)
            
            # Prepare features
            features = self.prepare_features(input_data)
            
            print("\nDebug - DataFrame head:")
            print(features.head())
            
            # Make prediction using scikit-learn interface
            prediction = model.predict(features)
            return float(prediction[0])
        except Exception as e:
            print("\nDebug - Input data:", input_data)
            print("\nDebug - Error:", str(e))
            raise ValueError(f"Error making prediction: {str(e)}")
