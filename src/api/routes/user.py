from fastapi import APIRouter, HTTPException, status

from api.models.schema import UserCreateRequest, UserResponse, UserUpdateRequest
from db.db_util import SupabaseManager

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreateRequest):
    try:
        manager = SupabaseManager()

        existing_user = await manager.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="User with this email already exists"
            )

        existing_user = await manager.get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this username already exists",
            )

        user = await manager.create_user(
            username=user_data.username, email=user_data.email, password=user_data.password
        )

        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            created_at=user["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, user_data: UserUpdateRequest):
    try:
        manager = SupabaseManager()

        # Check if user exists
        existing_user = await manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        # Update the user
        updated_user = await manager.update_user(
            user_id=user_id,
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
        )

        return UserResponse(
            id=updated_user["id"],
            username=updated_user["username"],
            email=updated_user["email"],
            created_at=updated_user["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}",
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    try:
        manager = SupabaseManager()

        existing_user = await manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        success = await manager.delete_user(user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete user"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}",
        )
