from fastapi import APIRouter, HTTPException, status

from api.models.schema import (
    HoldingCreateRequest,
    HoldingResponse,
    HoldingDeleteRequest,
)
from db.db_util import SupabaseManager

router = APIRouter(prefix="/holdings", tags=["holdings"])


@router.post("/", response_model=HoldingResponse, status_code=status.HTTP_201_CREATED)
async def create_holding(holding_data: HoldingCreateRequest):
    try:
        manager = SupabaseManager()
        holding = await manager.create_holding(
            user_id=holding_data.user_id, stock=holding_data.stock
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
async def delete_holding(holding_data: HoldingDeleteRequest):
    try:
        manager = SupabaseManager()
        success = await manager.delete_holding_by_user_and_stock(
            user_id=holding_data.user_id, stock=holding_data.stock
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


@router.get("/{user_id}", response_model=list[HoldingResponse])
async def get_user_holdings(user_id: str):
    try:
        manager = SupabaseManager()
        holdings = await manager.get_holdings_by_user(user_id)

        return [
            HoldingResponse(user_id=holding["user_id"], stock=holding["stock"])
            for holding in holdings
        ]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve holdings: {str(e)}",
        )
