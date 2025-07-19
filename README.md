# ArogyaAI: AI-Powered Vaccine Management Platform

ArogyaAI is an end-to-end platform that leverages artificial intelligence to help healthcare providers manage vaccine distribution, predict demand, identify dropout risks, and detect zero-dose clusters. The system consists of a modern React frontend and a FastAPI backend with integrated machine learning models and data files.

---

## 🚀 Main Features
- **Vaccine Demand Forecasting:** Predict short-term vaccine demand per district using LSTM models and climate data.
- **Dropout Risk Prediction:** Identify children at risk of missing vaccinations for early intervention.
- **Zero-Dose Cluster Detection:** Detect and prioritize high-risk areas for targeted vaccination efforts.
- **Beautiful, Responsive UI:** Modern, mobile-friendly interface built with React and Tailwind CSS.
- **Interactive API Docs:** Explore and test APIs via FastAPI's built-in Swagger UI.

---

## 🏗️ Architecture Overview

```
+-------------------+         REST API         +-------------------+
|    Frontend       | <---------------------> |     Backend       |
|  (React + Vite)   |                         |   (FastAPI, ML)   |
+-------------------+                         +-------------------+
         |                                              |
         |<----------->  ML Models, Data, etc. <--------|
```
- **Frontend:** User interface for data entry, visualization, and results.
- **Backend:** FastAPI server hosting ML models and API endpoints.
- **ML Models:** LSTM, XGBoost, KMeans, etc. for predictions and clustering.
- **Data:** CSVs and pickled models for predictions and analytics.

---
#Live Link : https://arogya-ai-nine.vercel.app/
# 📦 Project Structure
```
VaccineLedger/
  Frontend/    # React app (UI)
  Backend/     # FastAPI app (APIs, ML models, data)
  README.md    # (This file)
```

---

# 🖥️ Frontend (React + Vite)

## Features
- Modern, responsive UI with Tailwind CSS
- Pages: Home, Features, Forecasting, Dropout, Clustering, About, Mission, Contact
- Interactive forms for predictions
- Real-time API integration
- Themed layouts and mobile-first design

## Tech Stack
- React
- Vite
- Tailwind CSS
- Heroicons

## Setup & Scripts
```bash
cd Frontend
npm install         # Install dependencies
npm run dev         # Start development server (http://localhost:5173)
npm run build       # Build for production (output in dist/)
npm run lint        # Lint code
```

## Folder Structure
```
Frontend/
  src/
    components/    # Reusable UI components
    pages/         # Main app pages (Home, Features, Forecasting, Dropout, Clustering, etc.)
    assets/        # Images, icons, etc.
  public/          # Static files
  tailwind.config.js
  vite.config.js
  ...
```

---

# 🧠 Backend (FastAPI + ML)

## Features
- FastAPI server with modular routers
- ML models for forecasting, dropout, and clustering
- Pretrained models and data files included
- Interactive API docs at `/docs`

## Tech Stack
- Python 3.8+
- FastAPI
- Uvicorn
- scikit-learn, XGBoost, Keras
- Pandas, NumPy
- Docker (optional)

## Setup & Scripts
```bash
cd Backend
pip install -r requirements.txt   # Install dependencies
uvicorn main:app --reload         # Start API server (http://localhost:8000)
# Docker alternative:
docker build -t arogyaai-backend .
docker run -p 8000:8000 arogyaai-backend
```

## Folder Structure
```
Backend/
  main.py            # FastAPI app entry point
  routers/           # API route modules (forecasting, dropout, cluster)
  cluster_model/     # Pretrained clustering models (pkl)
  notebooks/         # Model training and analysis notebooks
  data/              # Datasets (CSV)
  requirements.txt   # Python dependencies
  ...
```

## Data & Models
- **/data/**: Contains CSVs for dropout prediction, demand forecasting, and clustering.
  - `dropout_prediction_satara.csv`: Data for dropout risk model
  - `vaccine_demand_forecasting.csv`: Data for demand forecasting
  - `zero_dose_clusters_maharashtra.csv`: Data for cluster detection
- **/cluster_model/**, **/notebooks/models/**, **/notebooks/vaccine_models/**: Pretrained ML models (KMeans, LSTM, etc.) in `.pkl` or `.h5` format
- **How it works:**
  - Models are loaded at API startup and used for predictions on incoming requests.
  - Data files are used for retraining, validation, and analytics.

## API Endpoints
- **/api/forecast/predict**  
  Predict vaccine demand (POST)
- **/api/dropout/predict**  
  Predict dropout risk (POST)
- **/api/cluster/predict**  
  Detect zero-dose clusters (POST)

See [http://localhost:8000/docs](http://localhost:8000/docs) for full API documentation and interactive testing.

---

# 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

# 📄 License
This project is licensed under the MIT License.

---

# 📬 Contact
For questions or support, please contact the project maintainer. 
