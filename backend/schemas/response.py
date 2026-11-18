from pydantic import BaseModel
from typing import List, Optional


class EvaluationResult(BaseModel):
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    log_loss: Optional[float] = None


class PredictionResponse(BaseModel):
    trip_id: str
    status: str
    predicted_route: List[int]
    evaluation: Optional[EvaluationResult] = None
    processing_time_seconds: Optional[float] = None

class PredictionRequest(BaseModel):
    trip_id: str

class TripListResponse(BaseModel):
    trips: List[str]
    total: int