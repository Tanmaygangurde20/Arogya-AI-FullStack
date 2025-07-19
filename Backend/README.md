# ArogyaAI Backend

ArogyaAI Backend is a FastAPI-based service that powers the AI-driven features of the ArogyaAI platform, including vaccine demand forecasting, dropout risk prediction, and zero-dose cluster detection.

## Features
- **Vaccine Demand Forecasting API:** Predicts short-term vaccine demand using LSTM models and climate data.
- **Dropout Risk Prediction API:** Identifies children at risk of missing vaccinations for early intervention.
- **Zero-Dose Cluster Detection API:** Detects and prioritizes high-risk areas for targeted vaccination efforts.
- **Modular, Scalable Design:** Built with FastAPI for high performance and easy extensibility.

## Tech Stack
- [Python 3.8+](https://www.python.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/) (ASGI server)
- [scikit-learn, XGBoost, Keras, etc.](https://scikit-learn.org/)

## Getting Started

### Prerequisites
- Python 3.8 or above
- pip

### Installation
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the API server:**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at [http://localhost:8000](http://localhost:8000)

### Docker (Optional)
You can also run the backend using Docker:
```bash
docker build -t arogyaai-backend .
docker run -p 8000:8000 arogyaai-backend
```

## Project Structure
```
Backend/
  main.py            # FastAPI app entry point
  routers/           # API route modules (forecasting, dropout, cluster)
  cluster_model/     # Pretrained clustering models
  notebooks/         # Model training and analysis notebooks
  data/              # Datasets
  requirements.txt   # Python dependencies
  ...
```

## API Endpoints
- **/api/forecast/predict**  
  Predict vaccine demand (POST)
- **/api/dropout/predict**  
  Predict dropout risk (POST)
- **/api/cluster/predict**  
  Detect zero-dose clusters (POST)

See the FastAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs) for full API details and interactive testing.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Contact
For questions or support, please contact the project maintainer.