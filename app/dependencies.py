# app/dependencies.py

from fastapi import Request, Response, HTTPException, status, Depends
from app.auth import create_access_token, verify_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

def get_session(request: Request, response: Response):
    token = request.cookies.get("access_token")
    if token:
        try:
            # Try to decode the token
            payload = verify_access_token(token)
            return payload
        except Exception:
            # Token is expired or invalid – fall through to create a new one.
            pass

    # Create a new session if none exists or the token is invalid.
    # Here we generate a unique session ID and set a default level (e.g., level 1).
    session_data = {"sid": str(uuid.uuid4()), "level": 1}
    token = create_access_token(session_data, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=os.getenv("DEBUG") == "False",  # Get from DEBUG env variable
    )
    return session_data
