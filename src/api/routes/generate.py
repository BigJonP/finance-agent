from fastapi import APIRouter

from api.models.schema import AdviceRequest
from api.services.advisor import generate_advice

router = APIRouter()


@router.post("/generate_advice")
def get_advice(request: AdviceRequest):
    return generate_advice(request)
