
from fastapi import FastAPI
from backend.routers.trips import router as trips_router
from backend.routers.predictions import router as predictions_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction System",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trips_router)
app.include_router(predictions_router)

@app.get("/")
def home():
    return {
        "message": "PathSnap API is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
