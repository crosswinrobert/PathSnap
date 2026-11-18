from fastapi import APIRouter, HTTPException

from backend.schemas.response import PredictionResponse

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post("/{trip_id}", response_model=PredictionResponse)
def predict_trip(trip_id: str):

    raise HTTPException(
        status_code=503,
        detail="Prediction pipeline is not connected yet."
    )