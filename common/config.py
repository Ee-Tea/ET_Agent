from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    APP_ENV: str = "local"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
