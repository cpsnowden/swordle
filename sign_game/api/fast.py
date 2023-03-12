from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sign_game.api.routers import letterprediction
from sign_game.api.config import Settings, get_settings
from sign_game.ml.registry import load_model

# https://fastapi.tiangolo.com/advanced/events/ - handle startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup")
    app.state.model = load_model(get_settings().current_model)
    yield
    print("Application shutdown")


app = FastAPI(lifespan=lifespan)
app.include_router(letterprediction.router)

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return { 'ping': 'pong' }

@app.get("/info")
async def info(settings: Settings = Depends(get_settings)):
    return {
        "current_model": settings.current_model
    }
