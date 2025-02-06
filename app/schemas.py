# schemas.py
from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    user_prompt: str
    max_new_tokens: int = Field(50, le=200, description="Maximum new tokens (capped at 200)")
