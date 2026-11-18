from fastapi import APIRouter, HTTPException

from backend.schemas.response import TripListResponse

router = APIRouter(
    prefix="/trips",
    tags=["Trips"]
)


@router.get("/", response_model=TripListResponse)
def get_trips():
    raise HTTPException(
        status_code=503,
        detail="Trip dataset has not been connected yet."
    )