from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sign_game.api.routers import letterprediction
from sign_game.ml.registry import load_model

app = FastAPI()

print("Loading Model")
app.state.model = load_model()
print("Loaded Model")

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
