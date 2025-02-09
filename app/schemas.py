# app/schemas.py

from pydantic import BaseModel, Field

class GenerationRequest(BaseModel):
    user_prompt: str
    max_new_tokens: int = Field(50, le=200, description="Maximum new tokens (capped at 200)")
    system_prompt_choice: str = Field("level-1", description="Key for the system prompt to use")

class ChangeModelRequest(BaseModel):
    model_uuid: str
