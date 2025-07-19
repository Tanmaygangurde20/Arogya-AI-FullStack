import requests
import json

# Test data for different patient scenarios
test_cases = [
    {
        "gender": "M",
        "age": 1,
        "travel_time": 20,
        "parent_education": "Graduate",
        "dose1_date": "2024-01-20",
        "dose2_date": "2024-02-15",
        "distance_to_center": 3.0,
        "delay_days": 26
    },
    {
        "gender": "F",
        "age": 2,
        "travel_time": 35,
        "parent_education": "Primary",
        "dose1_date": "2024-01-25",
        "dose2_date": "2024-03-10",
        "distance_to_center": 6.0,
        "delay_days": 45
    },
    {
        "gender": "M",
        "age": 1,
        "travel_time": 10,
        "parent_education": "Graduate",
        "dose1_date": "2024-01-30",
        "dose2_date": "2024-02-18",
        "distance_to_center": 1.5,
        "delay_days": 19
    },
    {
        "gender": "F",
        "age": 3,
        "travel_time": 45,
        "parent_education": "Secondary",
        "dose1_date": "2024-02-01",
        "dose2_date": "2024-03-20",
        "distance_to_center": 8.0,
        "delay_days": 48
    }
]

def test_dropout_api():
    #base_url = "http://localhost:8000/api/dropout"
    base_url = "https://Tanmay0483-ArogyaAI.hf.space/api/dropout/predict"
    
    print("🧪 Testing Vaccination Dropout Prediction API")
    print("=" * 60)
    
    # Test model info endpoint
    print("\n📋 Testing Model Info Endpoint:")
    print("-" * 40)
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model Info Retrieved Successfully!")
            if "model_info" in result:
                model_info = result["model_info"]
                print(f"🤖 Model: {model_info.get('model_name', 'Unknown')}")
                print(f"📅 Training Date: {model_info.get('training_date', 'Unknown')}")
                print(f"🎯 Best Score: {model_info.get('best_score', 'Unknown')}")
        else:
            print(f"❌ Error getting model info: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running!")
        print("   Run: uvicorn main:app --reload")
        return
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
    
    # Test prediction endpoint
    print("\n🎯 Testing Prediction Endpoint:")
    print("-" * 40)
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_data['gender']} child, age {test_data['age']}")
        print(f"Input: {json.dumps(test_data, indent=2)}")
        
        try:
            response = requests.post(f"{base_url}/predict", json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Status: {response.status_code}")
                
                if "error" not in result:
                    print(f"🎯 Prediction: {result['prediction_text']}")
                    print(f"📊 Confidence: {result['confidence']:.1f}%")
                    print(f"⚠️ Risk Level: {result['risk_level']}")
                    print(f"✅ Probability On Time: {result['probability_on_time']:.1f}%")
                    print(f"❌ Probability Delayed: {result['probability_delayed']:.1f}%")
                    
                    # Risk level color coding
                    risk_color = {
                        'Low': '🟢',
                        'Medium': '🟡', 
                        'High': '🔴'
                    }
                    print(f"{risk_color.get(result['risk_level'], '⚪')} Risk Assessment: {result['risk_level']}")
                else:
                    print(f"❌ API Error: {result['error']}")
            else:
                print(f"❌ HTTP Error! Status: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the FastAPI server is running!")
            print("   Run: uvicorn main:app --reload")
            break
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        print("-" * 50)

def test_single_prediction():
    """Test a single prediction with detailed output"""
    print("\n🎯 Single Prediction Test:")
    print("-" * 40)
    
    test_data = {
        "gender": "M",
        "age": 1,
        "travel_time": 20,
        "parent_education": "Graduate",
        "dose1_date": "2024-01-20",
        "dose2_date": "2024-02-15",
        "distance_to_center": 3.0,
        "delay_days": 26
    }
    
    try:
        response = requests.post("https://Tanmay0483-ArogyaAI.hf.space/api/dropout/predict", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction Successful!")
            print(f"📊 Full Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running!")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_dropout_api()
    
    # Run single prediction test
    test_single_prediction()
    
    print("\n" + "=" * 60)
    print("📋 API USAGE SUMMARY")
    print("=" * 60)
    print("""
🎯 Prediction Endpoint: POST /api/dropout/predict
📊 Model Info Endpoint: GET /api/dropout/model-info

📝 Expected Input Format:
{
    "gender": "M",
    "age": 1,
    "travel_time": 20,
    "parent_education": "Graduate",
    "dose1_date": "2024-01-20",
    "dose2_date": "2024-02-15",
    "distance_to_center": 3.0,
    "delay_days": 26
}

📊 Expected Output Format:
{
    "prediction_label": 1,
    "prediction_text": "On Time",
    "confidence": 99.9,
    "probability_delayed": 0.1,
    "probability_on_time": 99.9,
    "risk_level": "Low",
    "input_data": {...}
}
    """)
