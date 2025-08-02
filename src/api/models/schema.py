from pydantic import BaseModel


class AdviceRequest(BaseModel):
    user_id: str
    prompt: str


class AdviceResponse(BaseModel):
    advice: str
