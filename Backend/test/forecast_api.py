import requests
import json

# Test data for different district-vaccine combinations
test_cases = [
    {
        "district": "Pune",
        "vaccine_type": "BCG",
        "temperature": 22.5,
        "rainfall": 0.0,
        "stock_left": 30,
        "holiday_indicator": 0
    },
    {
        "district": "Mumbai",
        "vaccine_type": "Polio",
        "temperature": 35.0,
        "rainfall": 0.5,
        "stock_left": 50,
        "holiday_indicator": 0
    },
    {
        "district": "Nashik",
        "vaccine_type": "Measles",
        "temperature": 28.0,
        "rainfall": 0.0,
        "stock_left": 25,
        "holiday_indicator": 1
    }
]

def test_forecast_api():
   # base_url = "http://localhost:8000/api/forecast/predict"
    base_url = "https://tanmay0483-arogyaai.hf.space/api/forecast/predict"
    
    print("🧪 Testing Vaccine Forecast API")
    print("=" * 50)
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_data['district']} - {test_data['vaccine_type']}")
        print(f"Input: {json.dumps(test_data, indent=2)}")
        
        try:
            response = requests.post(base_url, json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Status: {response.status_code}")
                print(f"Response: {json.dumps(result, indent=2)}")
                
                if "prediction" in result:
                    print(f"🎯 Predicted Demand: {result['prediction']} doses")
                    print(f"🤖 Model Used: {result['model']}")
            else:
                print(f"❌ Error! Status: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the FastAPI server is running!")
            print("   Run: uvicorn main:app --reload")
            break
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_forecast_api()