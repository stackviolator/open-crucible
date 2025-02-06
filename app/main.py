from fastapi import FastAPI
from routes import router

app = FastAPI()

# Register routes from routes.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

