import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from io import StringIO
import warnings
from fastapi import APIRouter
from pydantic import BaseModel
warnings.filterwarnings('ignore')

router = APIRouter()

# Get the current directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to Backend, then to notebooks
NOTEBOOKS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'notebooks')

# Global variables with relative paths
MODELS_DIR = os.path.join(NOTEBOOKS_DIR, 'vaccine_models')
SCALERS_DIR = os.path.join(NOTEBOOKS_DIR, 'vaccine_scalers')

# Hardcoded historical data
HISTORICAL_DATA = """
Date,District,Vaccine Type,Administered Doses,Temperature,Rainfall,Stock Left,Holiday Indicator
2024-02-07,Mumbai,Polio,380,39.7,0.2,130,0
2024-02-07,Nashik,Measles,260,35.5,0.1,0,0
2024-02-08,Pune,BCG,310,37.2,0.0,0,0
2024-02-08,Mumbai,Polio,385,39.9,0.0,120,0
2024-02-08,Nashik,Measles,265,35.7,0.0,0,0
2024-02-09,Pune,BCG,315,37.4,0.0,0,0
2024-02-09,Mumbai,Polio,390,40.1,0.1,110,0
2024-02-09,Nashik,Measles,270,35.9,0.0,0,0
2024-02-10,Pune,BCG,320,37.6,0.1,0,1
2024-02-10,Mumbai,Polio,395,40.3,0.0,100,1
2024-02-10,Nashik,Measles,275,36.2,0.0,0,1
2024-02-11,Pune,BCG,325,37.9,0.0,0,0
2024-02-11,Mumbai,Polio,400,40.5,0.0,90,0
2024-02-11,Nashik,Measles,280,36.4,0.0,0,0
2024-02-12,Pune,BCG,330,38.0,0.1,0,0
2024-02-12,Mumbai,Polio,405,40.8,0.0,80,0
2024-02-12,Nashik,Measles,285,36.6,0.1,0,0
2024-02-13,Pune,BCG,335,38.2,0.0,0,0
2024-02-13,Mumbai,Polio,410,41.0,0.0,70,0
2024-02-13,Nashik,Measles,290,36.9,0.0,0,0
2024-02-14,Pune,BCG,340,38.5,0.0,0,0
2024-02-14,Mumbai,Polio,415,41.2,0.1,60,0
2024-02-14,Nashik,Measles,295,37.1,0.0,0,0
"""

# Load data into DataFrame
df = pd.read_csv(StringIO(HISTORICAL_DATA))
df['Date'] = pd.to_datetime(df['Date'])

# Model paths using relative paths
MODEL_PATHS = {
    ('Mumbai', 'Polio'): os.path.join(MODELS_DIR, 'Mumbai_Polio_model.keras'),
    ('Nashik', 'Measles'): os.path.join(MODELS_DIR, 'Nashik_Measles_model.keras'),
    ('Pune', 'BCG'): os.path.join(MODELS_DIR, 'Pune_BCG_model.keras'),
}

SCALER_PATHS = {
    ('Mumbai', 'Polio'): os.path.join(SCALERS_DIR, 'Mumbai_Polio_scaler.pkl'),
    ('Nashik', 'Measles'): os.path.join(SCALERS_DIR, 'Nashik_Measles_scaler.pkl'),
    ('Pune', 'BCG'): os.path.join(SCALERS_DIR, 'Pune_BCG_scaler.pkl'),
}

# Create dictionary for recent data (last 5 records per district-vaccine)
recent_data_dict = {}
for (district, vaccine), group in df.groupby(['District', 'Vaccine Type']):
    recent_data_dict[(district, vaccine)] = group.sort_values('Date').tail(5)

class ForecastInput(BaseModel):
    district: str
    vaccine_type: str
    temperature: float
    rainfall: float
    stock_left: int
    holiday_indicator: int

@router.post("/predict")
def forecast_predict(input: ForecastInput):
    """
    Predict vaccine demand for a specific district and vaccine type
    """
    # Check if the district-vaccine combination exists
    key = (input.district, input.vaccine_type)
    if key not in recent_data_dict:
        return {
            "error": f"No model available for {input.district} - {input.vaccine_type}",
            "available_combinations": [
                f"{district} - {vaccine}" for district, vaccine in recent_data_dict.keys()
            ]
        }
    
    try:
        # Load model and scaler
        model_path = MODEL_PATHS.get(key)
        scaler_path = SCALER_PATHS.get(key)
        
        if not model_path or not scaler_path:
            return {
                "error": f"Model files not found for {input.district} - {input.vaccine_type}"
            }
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return {
                "error": f"Model or scaler file does not exist for {input.district} - {input.vaccine_type}",
                "model_path": model_path,
                "scaler_path": scaler_path
            }
        
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Get recent data for this combination
        recent_data = recent_data_dict[key]
        
        # Prepare features
        features = ['Administered Doses', 'Temperature', 'Rainfall', 'Stock Left', 'Holiday Indicator']
        input_data = recent_data[features].values.copy()
        
        # Update last row with new inputs
        input_data[-1, 1] = input.temperature
        input_data[-1, 2] = input.rainfall
        input_data[-1, 3] = input.stock_left
        input_data[-1, 4] = input.holiday_indicator
        
        # Scale and predict
        scaled_input = scaler.transform(input_data)
        lstm_input = scaled_input.reshape(1, 5, len(features))
        scaled_prediction = model.predict(lstm_input, verbose=0)
        
        # Inverse transform
        dummy = np.zeros((1, len(features)))
        dummy[0, 0] = scaled_prediction[0, 0]
        actual_prediction = scaler.inverse_transform(dummy)[0, 0]
        prediction = max(0, int(actual_prediction))
        
        return {
            "model": "LSTM",
            "prediction": prediction,
            "district": input.district,
            "vaccine_type": input.vaccine_type,
            "input_parameters": {
                "temperature": input.temperature,
                "rainfall": input.rainfall,
                "stock_left": input.stock_left,
                "holiday_indicator": input.holiday_indicator
            }
        }
        
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "district": input.district,
            "vaccine_type": input.vaccine_type
        }

def predict_demand(input_dict):
    # Validate input
    required_keys = ['temperature', 'rainfall', 'stock_left', 'holiday']
    for key in required_keys:
        if key not in input_dict:
            return {'error': f"Missing required input: {key}"}
    
    # Extract inputs
    temperature = float(input_dict['temperature'])
    rainfall = float(input_dict['rainfall'])
    stock_left = int(input_dict['stock_left'])
    holiday = int(input_dict['holiday'])
    
    results = {}
    
    # Process each district-vaccine combination
    for (district, vaccine_type), recent_data in recent_data_dict.items():
        key = f"{district}_{vaccine_type}"
        
        try:
            # Load model and scaler
            model_path = MODEL_PATHS.get((district, vaccine_type))
            scaler_path = SCALER_PATHS.get((district, vaccine_type))
            
            # Check if files exist
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                prediction = "Model files not found"
            else:
                model = load_model(model_path)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # Prepare features
                features = ['Administered Doses', 'Temperature', 'Rainfall', 'Stock Left', 'Holiday Indicator']
                input_data = recent_data[features].values.copy()
                
                # Update last row with new inputs
                input_data[-1, 1] = temperature
                input_data[-1, 2] = rainfall
                input_data[-1, 3] = stock_left
                input_data[-1, 4] = holiday
                
                # Scale and predict
                scaled_input = scaler.transform(input_data)
                lstm_input = scaled_input.reshape(1, 5, len(features))
                scaled_prediction = model.predict(lstm_input, verbose=0)
                
                # Inverse transform
                dummy = np.zeros((1, len(features)))
                dummy[0, 0] = scaled_prediction[0, 0]
                actual_prediction = scaler.inverse_transform(dummy)[0, 0]
                prediction = max(0, int(actual_prediction))
            
        except Exception as e:
            prediction = f"Model not available: {str(e)}"
        
        results[key] = {
            'district': district,
            'vaccine_type': vaccine_type,
            'predicted_demand': prediction
        }
    
    return results

# Simple usage function (remains same)
def get_predictions(temperature, rainfall, stock_left, holiday):
    input_data = {
        'temperature': temperature,
        'rainfall': rainfall,
        'stock_left': stock_left,
        'holiday': holiday
    }
    return predict_demand(input_data)

if __name__ == "__main__":
    # Example usage
    predictions = get_predictions(
        temperature=25.0,
        rainfall=2.5,
        stock_left=10,
        holiday=0
    )
    
    # Print results
    if 'error' in predictions:
        print(f"Error: {predictions['error']}")
    else:
        print("Vaccine Demand Predictions:")
        print("-" * 50)
        for key, pred in predictions.items():
            district = pred['district']
            vaccine = pred['vaccine_type']
            demand = pred['predicted_demand']
            print(f"{district:10} | {vaccine:8} | {demand}")
    
    # Print paths for debugging
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Notebooks directory: {NOTEBOOKS_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Scalers directory: {SCALERS_DIR}")
    print("\nModel paths:")
    for key, path in MODEL_PATHS.items():
        print(f"  {key}: {path}")
    print("\nScaler paths:")
    for key, path in SCALER_PATHS.items():
        print(f"  {key}: {path}")