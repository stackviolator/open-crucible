# app/dependencies.py

from fastapi import Request, Response, HTTPException, status, Depends
from app.auth import create_access_token, verify_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

def get_session(request: Request):
    token = request.cookies.get("access_token")
    if token:
        try:
            # Try to decode the token.
            payload = verify_access_token(token)
            return payload
        except Exception:
            # Token is expired or invalid.
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    # No token was provided.
    raise HTTPException(status_code=401, detail="Authentication required")

def internal_only(request: Request):
    """
    Dependency that restricts access to internal endpoints.
    Checks if the request originates from localhost.
    """
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="This endpoint is only accessible internally.")
