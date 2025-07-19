import requests
import json

# Test data for different types of zero-dose clusters
test_cases = [
    {
        "area_id": "AREA_001",
        "city_name": "Mumbai",
        "district_name": "Mumbai City",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "zero_dose_count": 200,
        "income": 30000,
        "travel_time": 45,
        "literacy_rate": 65.0
    },
    {
        "area_id": "AREA_002", 
        "city_name": "Pune",
        "district_name": "Pune",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "zero_dose_count": 80,
        "income": 75000,
        "travel_time": 25,
        "literacy_rate": 85.0
    },
    {
        "area_id": "AREA_003",
        "city_name": "Nashik",
        "district_name": "Nashik",
        "latitude": 19.9975,
        "longitude": 73.7898,
        "zero_dose_count": 120,
        "income": 45000,
        "travel_time": 70,
        "literacy_rate": 55.0
    },
    {
        "area_id": "AREA_004",
        "city_name": "Nagpur",
        "district_name": "Nagpur",
        "latitude": 21.1458,
        "longitude": 79.0882,
        "zero_dose_count": 40,
        "income": 90000,
        "travel_time": 15,
        "literacy_rate": 90.0
    },
    {
        "area_id": "AREA_005",
        "city_name": "Aurangabad",
        "district_name": "Aurangabad",
        "latitude": 19.8762,
        "longitude": 75.3433,
        "zero_dose_count": 180,
        "income": 25000,
        "travel_time": 90,
        "literacy_rate": 45.0
    }
]

def test_cluster_api():
    #base_url = "http://localhost:8000/api/cluster"
    base_url = "https://tanmay0483-arogyaai.hf.space/api/cluster"
    
    print("🔍 Testing Zero-Dose Cluster Detection API")
    print("=" * 70)
    
    for i, test_data in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_data['city_name']} - {test_data['district_name']}")
        print(f"📍 Coordinates: ({test_data['latitude']}, {test_data['longitude']})")
        print(f"📊 Zero-dose Count: {test_data['zero_dose_count']}")
        print(f"💰 Income: ₹{test_data['income']:,}")
        print(f"⏱️ Travel Time: {test_data['travel_time']} minutes")
        print(f"📚 Literacy Rate: {test_data['literacy_rate']}%")
        
        try:
            response = requests.post(f"{base_url}/predict", json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Status: {response.status_code}")
                
                if "error" not in result:
                    # Display cluster information
                    print(f"\n🎯 Cluster Analysis Results:")
                    print(f"   Cluster ID: {result['cluster_id']}")
                    print(f"   Cluster Type: {result['cluster_type']}")
                    print(f"   Risk Level: {result['risk_level']}")
                    print(f"   Intervention Priority: {result['intervention_priority']}")
                    
                    # Display current metrics
                    print(f"\n📊 Current Area Metrics:")
                    current = result['current_metrics']
                    print(f"   Zero-dose Count: {current['zero_dose_count']}")
                    print(f"   Income: ₹{current['income']:,}")
                    print(f"   Travel Time: {current['travel_time']} minutes")
                    print(f"   Literacy Rate: {current['literacy_rate']}%")
                    print(f"   Priority Score: {current['priority_score']}")
                    
                    # Display cluster characteristics
                    print(f"\n📈 Cluster Characteristics (Average):")
                    cluster_chars = result['cluster_characteristics']
                    print(f"   Avg Zero-dose: {cluster_chars['avg_zero_dose']}")
                    print(f"   Avg Income: ₹{cluster_chars['avg_income']:,}")
                    print(f"   Avg Travel Time: {cluster_chars['avg_travel_time']} minutes")
                    print(f"   Avg Literacy: {cluster_chars['avg_literacy']}%")
                    print(f"   Avg Priority Score: {cluster_chars['avg_priority_score']}")
                    print(f"   Similar Areas: {cluster_chars['similar_areas']}")
                    
                    # Display recommendations
                    print(f"\n💡 Recommendations:")
                    for j, rec in enumerate(result['recommendations'], 1):
                        print(f"   {j}. {rec}")
                    
                    # Risk level color coding
                    risk_colors = {
                        'Critical': '🔴',
                        'High': '🟠',
                        'Medium': '🟡',
                        'Low': '🟢'
                    }
                    print(f"\n{risk_colors.get(result['risk_level'], '⚪')} Risk Assessment: {result['risk_level']}")
                    
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
        
        print("-" * 70)

def test_single_cluster():
    """Test a single cluster prediction with detailed output"""
    print("\n🎯 Single Cluster Prediction Test:")
    print("-" * 50)
    
    test_data = {
        "area_id": "AREA_TEST",
        "city_name": "Mumbai",
        "district_name": "Mumbai City",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "zero_dose_count": 200,
        "income": 30000,
        "travel_time": 45,
        "literacy_rate": 65.0
    }
    
    try:
        response = requests.post("http://localhost:8000/api/cluster/predict", json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Cluster Prediction Successful!")
            print(f"📊 Full Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the FastAPI server is running!")
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

def test_cluster_comparison():
    """Compare different cluster types"""
    print("\n📊 Cluster Type Comparison:")
    print("-" * 50)
    
    # Test high-risk vs low-risk areas
    high_risk = {
        "area_id": "HIGH_RISK",
        "city_name": "High Risk Area",
        "district_name": "Test District",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "zero_dose_count": 250,
        "income": 20000,
        "travel_time": 90,
        "literacy_rate": 40.0
    }
    
    low_risk = {
        "area_id": "LOW_RISK",
        "city_name": "Low Risk Area", 
        "district_name": "Test District",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "zero_dose_count": 30,
        "income": 100000,
        "travel_time": 15,
        "literacy_rate": 95.0
    }
    
    test_cases = [("High Risk", high_risk), ("Low Risk", low_risk)]
    
    for case_name, test_data in test_cases:
        try:
            response = requests.post("http://localhost:8000/api/cluster/predict", json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    print(f"\n{case_name}:")
                    print(f"  Cluster Type: {result['cluster_type']}")
                    print(f"  Risk Level: {result['risk_level']}")
                    print(f"  Priority: {result['intervention_priority']}")
                    print(f"  Zero-dose Count: {result['current_metrics']['zero_dose_count']}")
                    print(f"  Income: ₹{result['current_metrics']['income']:,}")
                else:
                    print(f"{case_name}: Error - {result['error']}")
            else:
                print(f"{case_name}: HTTP Error {response.status_code}")
                
        except Exception as e:
            print(f"{case_name}: Exception - {str(e)}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_cluster_api()
    
    # Run single prediction test
    test_single_cluster()
    
    # Run comparison test
    test_cluster_comparison()
    
    print("\n" + "=" * 70)
    print("📋 CLUSTER API USAGE SUMMARY")
    print("=" * 70)
    print("""
🎯 Prediction Endpoint: POST /api/cluster/predict

📝 Expected Input Format:
{
    "area_id": "AREA_001",
    "city_name": "Mumbai",
    "district_name": "Mumbai City", 
    "latitude": 19.0760,
    "longitude": 72.8777,
    "zero_dose_count": 200,
    "income": 30000,
    "travel_time": 45,
    "literacy_rate": 65.0
}

📊 Expected Output Format:
{
    "cluster_id": 0,
    "cluster_type": "High-Risk Zero-Dose Cluster",
    "risk_level": "Critical",
    "area_info": {...},
    "current_metrics": {...},
    "cluster_characteristics": {...},
    "recommendations": [...],
    "intervention_priority": "Immediate"
}

🔍 Cluster Types:
- High-Risk Zero-Dose Cluster (Critical)
- Accessibility-Challenged Cluster (High)  
- Low-Literacy High-Dropout Cluster (High)
- Low-Risk Well-Served Cluster (Low)
- Moderate-Risk Cluster (Medium)

💡 Intervention Priorities:
- Immediate (>80 priority score)
- High (60-80 priority score)
- Medium (40-60 priority score)  
- Low (<40 priority score)
    """) 