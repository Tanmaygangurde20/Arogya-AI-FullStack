from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import forecasting, dropout, cluster

app = FastAPI(title="VaccineAI Backend API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(forecasting.router, prefix="/api/forecast", tags=["Forecasting"])
app.include_router(dropout.router, prefix="/api/dropout", tags=["Dropout"])
app.include_router(cluster.router, prefix="/api/cluster", tags=["Cluster"])

@app.get("/")
def root():
    return {"message": "VaccineAI Backend API"} 