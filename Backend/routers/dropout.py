# Placeholder for dropout router 

# Vaccination Timing Prediction System
# Use this code to make predictions with your saved model

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class DropoutInput(BaseModel):
    gender: str
    age: int
    travel_time: int
    parent_education: str
    dose1_date: str
    dose2_date: str
    distance_to_center: float
    delay_days: int

class VaccinationPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the prediction system
        
        Args:
            model_path (str): Path to the directory containing saved model files
        """
        if model_path is None:
            base_path = os.path.dirname(__file__)  # this file's path
            model_path = os.path.join(base_path, "../notebooks/models/")
        
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.metadata = None

    def load_model(self):
        """Load the saved model and all preprocessing components"""
        try:
            print("Loading model components...")
            
            metadata_file = os.path.join(self.model_path, "model_metadata.pkl")
            if os.path.exists(metadata_file):
                self.metadata = joblib.load(metadata_file)
                print(f"✅ Model: {self.metadata['best_model_name']}")
                print(f"✅ Training Date: {self.metadata['training_date']}")
                print(f"✅ Best Score: {self.metadata['best_score']:.4f}")
            
            model_files = [f for f in os.listdir(self.model_path) 
                          if f.startswith('best_model_') and f.endswith('.pkl')]
            if not model_files:
                raise FileNotFoundError("No saved model found in the specified path")
            
            model_file = os.path.join(self.model_path, model_files[0])
            self.model = joblib.load(model_file)
            print(f"✅ Model loaded from: {model_file}")
            
            self.scaler = joblib.load(os.path.join(self.model_path, "scaler.pkl"))
            self.label_encoders = joblib.load(os.path.join(self.model_path, "label_encoders.pkl"))
            self.feature_columns = joblib.load(os.path.join(self.model_path, "feature_columns.pkl"))
            
            print("✅ All components loaded")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def prepare_input_data(self, input_data):
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()
        
        if 'Dose1 Date' in df.columns:
            df['Dose1_Date'] = pd.to_datetime(df['Dose1 Date'])
        if 'Dose2 Date' in df.columns:
            df['Dose2_Date'] = pd.to_datetime(df['Dose2 Date'])
        
        if 'Dose1_Date' in df.columns:
            df['Dose1_Month'] = df['Dose1_Date'].dt.month
            df['Dose1_DayOfWeek'] = df['Dose1_Date'].dt.dayofweek
        
        if 'Dose1_Date' in df.columns and 'Dose2_Date' in df.columns:
            df['Days_Between_Doses'] = (df['Dose2_Date'] - df['Dose1_Date']).dt.days
        
        if 'Travel Time' in df.columns and 'Distance to Center' in df.columns:
            df['Travel_Distance_Ratio'] = df['Travel Time'] / df['Distance to Center']
            df['Travel_Distance_Ratio'] = df['Travel_Distance_Ratio'].replace([np.inf, -np.inf], 0)
        
        if 'Age' in df.columns and 'Travel Time' in df.columns:
            df['Age_Travel_Interaction'] = df['Age'] * df['Travel Time']
        
        if 'Gender' in df.columns:
            df['Gender_Encoded'] = self.label_encoders['Gender'].transform(df['Gender'])
        
        if 'Parent Education' in df.columns:
            education_mapping = {'Primary': 1, 'Secondary': 2, 'Graduate': 3}
            df['Parent_Education_Encoded'] = df['Parent Education'].map(education_mapping)
        
        X = df[self.feature_columns]
        X = X.fillna(X.mean())
        return X

    def predict(self, input_data, return_probabilities=False):
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        X = self.prepare_input_data(input_data)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions

    def predict_single(self, input_data):
        predictions, probabilities = self.predict(input_data, return_probabilities=True)
        result = {
            'prediction_label': int(predictions[0]),
            'prediction_text': 'On Time' if predictions[0] == 1 else 'Delayed',
            'confidence': float(max(probabilities[0]) * 100),
            'probability_delayed': float(probabilities[0][0] * 100),
            'probability_on_time': float(probabilities[0][1] * 100),
            'risk_level': 'Low' if probabilities[0][1] > 0.7 else 'Medium' if probabilities[0][1] > 0.4 else 'High'
        }
        return result

    def get_model_info(self):
        if self.metadata:
            return {
                'model_name': self.metadata['best_model_name'],
                'training_date': self.metadata['training_date'],
                'best_score': self.metadata['best_score'],
                'feature_columns': self.feature_columns,
                'model_type': self.metadata.get('model_type', 'Unknown')
            }
        return None

# Initialize predictor globally
predictor = VaccinationPredictor()

@router.post("/predict")
def dropout_predict(input: DropoutInput):
    try:
        if predictor.model is None:
            if not predictor.load_model():
                return {
                    "error": "Failed to load prediction model. Please check model files.",
                    "status": "error"
                }
        
        input_dict = {
            'Gender': input.gender,
            'Age': input.age,
            'Travel Time': input.travel_time,
            'Parent Education': input.parent_education,
            'Dose1 Date': input.dose1_date,
            'Dose2 Date': input.dose2_date,
            'Distance to Center': input.distance_to_center,
            'Delay_Days': input.delay_days
        }
        
        result = predictor.predict_single(input_dict)
        result['input_data'] = input_dict
        return result
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "status": "error",
            "input_data": {
                'Gender': input.gender,
                'Age': input.age,
                'Travel Time': input.travel_time,
                'Parent Education': input.parent_education,
                'Dose1 Date': input.dose1_date,
                'Dose2 Date': input.dose2_date,
                'Distance to Center': input.distance_to_center,
                'Delay_Days': input.delay_days
            }
        }

@router.get("/model-info")
def get_model_info():
    try:
        if predictor.model is None:
            if not predictor.load_model():
                return {
                    "error": "Failed to load prediction model",
                    "status": "error"
                }
        
        model_info = predictor.get_model_info()
        if model_info:
            return {
                "status": "success",
                "model_info": model_info
            }
        else:
            return {
                "error": "Model information not available",
                "status": "error"
            }
    except Exception as e:
        return {
            "error": f"Failed to get model info: {str(e)}",
            "status": "error"
        }
