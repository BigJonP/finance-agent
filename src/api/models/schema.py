from pydantic import BaseModel, EmailStr
from typing import Optional


class AdviceRequest(BaseModel):
    user_id: str
    prompt: str


class AdviceResponse(BaseModel):
    advice: str


class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str


class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
