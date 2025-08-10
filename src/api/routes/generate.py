from fastapi import APIRouter, Depends

from api.models.schema import AdviceResponse
from api.auth.jwt_handler import get_current_user
from api.services.advisor import generate_advice

router = APIRouter()


@router.post("/generate_advice", response_model=AdviceResponse)
async def get_advice(current_user: dict = Depends(get_current_user)) -> AdviceResponse:
    # Create a mock request with user_id from JWT token
    from api.models.schema import AdviceRequest

    request = AdviceRequest(user_id=current_user["user_id"])
    return await generate_advice(request)
