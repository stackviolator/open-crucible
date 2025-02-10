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
            response = Response(status_code=status.HTTP_303_SEE_OTHER)
            response.headers["Location"] = "/"
            return response
    # No token was provided.
    response = Response(status_code=status.HTTP_303_SEE_OTHER)
    response.headers["Location"] = "/"
    return response