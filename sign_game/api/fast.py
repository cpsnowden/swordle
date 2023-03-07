from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sign_game.api.routers import letterprediction

app = FastAPI()

app.include_router(letterprediction.router)

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/ping")
def root():
    return { 'ping': 'pong' }
