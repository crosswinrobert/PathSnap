
from fastapi import FastAPI

app = FastAPI(
    title="PathSnap API",
    description="GPS Map Matching and Route Reconstruction System",
    version="1.0.0"
)


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
