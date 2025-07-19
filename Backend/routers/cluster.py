from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import joblib
import os

router = APIRouter()

# Base path for models (relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../notebooks/cluster_model")

class ClusterInput(BaseModel):
    area_id: str
    city_name: str
    district_name: str
    latitude: float
    longitude: float
    zero_dose_count: int
    income: int
    travel_time: int
    literacy_rate: float

@router.post("/predict")
def cluster_predict(input: ClusterInput):
    input_dict = {
        'Area ID': input.area_id,
        'City Name': input.city_name,
        'District Name': input.district_name,
        'Latitude': input.latitude,
        'Longitude': input.longitude,
        'Zero-dose Count': input.zero_dose_count,
        'Income': input.income,
        'Travel Time': input.travel_time,
        'Literacy Rate': input.literacy_rate
    }
    result = predict_cluster(input_dict)
    return result

def predict_cluster(input_data):
    try:
        input_df = pd.DataFrame([input_data])

        input_df['Dose_Density'] = input_df['Zero-dose Count'] / (input_df['Income'] / 1000)
        input_df['Accessibility_Score'] = input_df['Literacy Rate'] / input_df['Travel Time']
        input_df['Priority_Score'] = (
            input_df['Zero-dose Count'] * 0.4 +
            input_df['Travel Time'] * 0.3 +
            (100 - input_df['Literacy Rate']) * 0.3
        )

        # Load models using relative path
        scaler_path = os.path.join(MODEL_DIR, "vaccination_scaler.pkl")
        model_path = os.path.join(MODEL_DIR, "vaccination_cluster_kmeans.pkl")
        summary_path = os.path.join(MODEL_DIR, "cluster_summary.pkl")

        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        cluster_summary = joblib.load(summary_path)

        features = [
            'Latitude', 'Longitude', 'Zero-dose Count', 'Income',
            'Travel Time', 'Literacy Rate', 'Dose_Density',
            'Accessibility_Score', 'Priority_Score'
        ]
        processed_input = scaler.transform(input_df[features])
        cluster = model.predict(processed_input)[0]

        cluster_profile = cluster_summary[cluster_summary['KMeans_Cluster'] == cluster].iloc[0]

        cluster_type, risk_level, recommendations = analyze_cluster(cluster, cluster_profile, input_data)

        return {
            'cluster_id': int(cluster),
            'cluster_type': cluster_type,
            'risk_level': risk_level,
            'area_info': {
                'area_id': input_data['Area ID'],
                'city_name': input_data['City Name'],
                'district_name': input_data['District Name'],
                'coordinates': {
                    'latitude': input_data['Latitude'],
                    'longitude': input_data['Longitude']
                }
            },
            'current_metrics': {
                'zero_dose_count': input_data['Zero-dose Count'],
                'income': input_data['Income'],
                'travel_time': input_data['Travel Time'],
                'literacy_rate': input_data['Literacy Rate'],
                'priority_score': round(input_df['Priority_Score'].iloc[0], 1)
            },
            'cluster_characteristics': {
                'avg_zero_dose': round(cluster_profile['Zero-dose Count'], 1),
                'avg_income': round(cluster_profile['Income'], 1),
                'avg_travel_time': round(cluster_profile['Travel Time'], 1),
                'avg_literacy': round(cluster_profile['Literacy Rate'], 1),
                'avg_priority_score': round(cluster_profile['Priority_Score'], 1),
                'similar_areas': cluster_profile['City Name']
            },
            'recommendations': recommendations,
            'intervention_priority': get_intervention_priority(cluster, input_data)
        }
    except Exception as e:
        return {
            'error': f'Prediction failed: {str(e)}',
            'status': 'error'
        }

def analyze_cluster(cluster, cluster_profile, input_data):
    avg_zero_dose = cluster_profile['Zero-dose Count']
    avg_income = cluster_profile['Income']
    avg_travel_time = cluster_profile['Travel Time']
    avg_literacy = cluster_profile['Literacy Rate']

    if avg_zero_dose > 150 and avg_income < 50000:
        cluster_type = "High-Risk Zero-Dose Cluster"
        risk_level = "Critical"
    elif avg_zero_dose > 100 and avg_travel_time > 60:
        cluster_type = "Accessibility-Challenged Cluster"
        risk_level = "High"
    elif avg_literacy < 70 and avg_zero_dose > 80:
        cluster_type = "Low-Literacy High-Dropout Cluster"
        risk_level = "High"
    elif avg_zero_dose < 50 and avg_income > 80000:
        cluster_type = "Low-Risk Well-Served Cluster"
        risk_level = "Low"
    else:
        cluster_type = "Moderate-Risk Cluster"
        risk_level = "Medium"

    recommendations = generate_recommendations(cluster_type, risk_level, input_data)
    return cluster_type, risk_level, recommendations

def generate_recommendations(cluster_type, risk_level, input_data):
    recommendations = []

    if "High-Risk Zero-Dose" in cluster_type:
        recommendations.extend([
            "🚨 Immediate mobile vaccination camps deployment",
            "💰 Financial incentives for vaccination completion",
            "🏥 Establish temporary vaccination centers",
            "📱 Intensive community outreach programs",
            "🎯 Door-to-door vaccination campaigns"
        ])
    elif "Accessibility-Challenged" in cluster_type:
        recommendations.extend([
            "🚐 Mobile vaccination units with extended hours",
            "🏠 Home-based vaccination services",
            "🚗 Transportation assistance programs",
            "📞 Telemedicine consultation services",
            "🗺️ Optimize vaccination center locations"
        ])
    elif "Low-Literacy" in cluster_type:
        recommendations.extend([
            "📚 Educational campaigns in local languages",
            "👥 Community health worker training",
            "📺 Radio and TV awareness programs",
            "🏫 School-based vaccination programs",
            "🤝 Religious leader engagement"
        ])
    elif "Low-Risk" in cluster_type:
        recommendations.extend([
            "✅ Maintain current vaccination programs",
            "📊 Regular monitoring and surveillance",
            "🎓 Continue health education initiatives",
            "🏆 Recognition programs for high coverage",
            "📈 Share best practices with other areas"
        ])
    else:
        recommendations.extend([
            "📋 Targeted awareness campaigns",
            "🏥 Strengthen existing health infrastructure",
            "📱 Digital appointment booking systems",
            "👨‍⚕️ Additional healthcare worker training",
            "📊 Enhanced data collection and monitoring"
        ])

    return recommendations

def get_intervention_priority(cluster, input_data):
    priority_score = (
        input_data['Zero-dose Count'] * 0.4 +
        input_data['Travel Time'] * 0.3 +
        (100 - input_data['Literacy Rate']) * 0.3
    )
    if priority_score > 80:
        return "Immediate"
    elif priority_score > 60:
        return "High"
    elif priority_score > 40:
        return "Medium"
    else:
        return "Low"
