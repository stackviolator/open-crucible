# main.py
import logging
from fastapi import FastAPI
from routes import router
from fastapi.staticfiles import StaticFiles

# Configure logging to output to both the console and a file ("app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),        # Logs to console
        logging.FileHandler("app.log")    # Logs to file "app.log"
    ]
)

app = FastAPI()

# Mount the static directory for JS/CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routes from routes.py
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
