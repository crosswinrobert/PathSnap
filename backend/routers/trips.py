
import os

from fastapi import APIRouter, HTTPException

from backend.schemas.response import TripListResponse
from src.data_io import list_trips

PARTITION = os.getenv("PATHSNAP_PARTITION", "student_test").strip()

router = APIRouter(
    prefix="/trips",
    tags=["Trips"]
)


@router.get("/", response_model=TripListResponse)
def get_trips():
    try:
        trips = list_trips(PARTITION)

        return TripListResponse(
            trips=trips,
            total=len(trips)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load trips: {str(exc)}"
        )
