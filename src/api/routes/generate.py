from fastapi import APIRouter

from api.models.schema import AdviceRequest, AdviceResponse
from api.services.advisor import generate_advice

router = APIRouter()


@router.post("/generate_advice", response_model=AdviceResponse)
async def get_advice(request: AdviceRequest) -> AdviceResponse:
    return await generate_advice(request)
