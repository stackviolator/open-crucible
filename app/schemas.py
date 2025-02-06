# schemas.py

from pydantic import BaseModel

class GenerationRequest(BaseModel):
    user_prompt: str
    max_new_tokens: int = 50

