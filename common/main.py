from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from .db import get_db, engine
from .models.base import Base
from .models.user import User, OAuthAccount
from .auth_routes import router as auth_router
from .config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting authentication service...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down authentication service...")

# Create FastAPI app
app = FastAPI(
    title="ET Agent Authentication API",
    description="Authentication service for ET Agent with Google OAuth",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication routes
app.include_router(auth_router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ET Agent Authentication API",
        "version": "1.0.0",
        "docs": "/docs",
        "auth_endpoints": {
            "google_auth": "/auth/google",
            "google_callback": "/auth/google/callback",
            "me": "/auth/me",
            "refresh": "/auth/refresh",
            "logout": "/auth/logout"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "et-agent-auth"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

