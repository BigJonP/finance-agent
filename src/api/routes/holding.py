from fastapi import APIRouter, HTTPException, status, Depends

from api.models.schema import (
    HoldingCreateRequest,
    HoldingResponse,
    HoldingDeleteRequest,
)
from api.auth.jwt_handler import get_current_user
from db.db_util import SupabaseManager

router = APIRouter(prefix="/holding", tags=["holding"])


@router.post("/", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
async def create_holding(
    holding_data: HoldingCreateRequest, current_user: dict = Depends(get_current_user)
):
    try:
        manager = SupabaseManager()
        holding = await manager.create_holding(
            user_id=current_user["user_id"], stock=holding_data.stock
        )

        return HoldingResponse(user_id=holding["user_id"], stock=holding["stock"])

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create holding: {str(e)}",
        )


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_holding(
    holding_data: HoldingDeleteRequest, current_user: dict = Depends(get_current_user)
):
    try:
        manager = SupabaseManager()
        success = await manager.delete_holding_by_user_and_stock(
            user_id=current_user["user_id"], stock=holding_data.stock
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete holding",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete holding: {str(e)}",
        )


@router.get("/", response_model=list[HoldingResponse])
async def get_user_holdings(current_user: dict = Depends(get_current_user)):
    try:
        manager = SupabaseManager()
        holdings = await manager.get_holdings_by_user(current_user["user_id"])

        return [
            HoldingResponse(user_id=holding["user_id"], stock=holding["stock"])
            for holding in holdings
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve holdings: {str(e)}",
        )
