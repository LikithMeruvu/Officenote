from fastapi import FastAPI
import logging
from fastapi.middleware.cors import CORSMiddleware

# Import routers from the routes package
from routes.officenote import router as officenote_router
from routes.news import router as news_router
from routes.WhatsMissing import router as whatsmissing_router
from routes.dbcheck import router as dbcheck_router
from routes.observations import router as observations_router

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

# Configure logging with debug level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Include the routers
app.include_router(officenote_router)
app.include_router(news_router)
app.include_router(dbcheck_router)
app.include_router(whatsmissing_router)
app.include_router(observations_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the DRHP Project API"}

@app.get("/test")
def test():
    return {"message": "Test endpoint is working!"}