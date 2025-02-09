from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse
from app.dependencies import get_session

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Paths that don't require authentication
        public_paths = {"/", "/login", "/login/github", "/auth/github", "/register", "/login/native", "/static"}
        
        # Check if the path is public
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)
            
        try:
            # Validate the token using get_session
            session = get_session(request)
            if not session:
                return RedirectResponse(url="/login", status_code=303)
        except HTTPException:
            return RedirectResponse(url="/login", status_code=303)
            
        response = await call_next(request)
        return response

def add_middleware(app: FastAPI):
    """Add all middleware to the application."""
    app.add_middleware(AuthMiddleware) 