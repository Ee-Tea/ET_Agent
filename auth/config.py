from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    APP_ENV: str = "local"
    
    # OAuth Settings
    GOOGLE_CLIENT_ID=REDACTED
    GOOGLE_CLIENT_SECRET=REDACTED
    GOOGLE_REDIRECT_URI: str
    
    # JWT Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Settings
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
