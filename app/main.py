# main.py

from fastapi import FastAPI
from routes import router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the static directory for JS/CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routes from routes.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
