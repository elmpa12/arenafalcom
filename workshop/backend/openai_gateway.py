"""FastAPI gateway exposing GPT code-generation capabilities.

This module expects an ``.env`` file (or environment variables) providing
``OPENAI_API_KEY``. Optionally, set ``GATEWAY_TOKEN`` to require an authentication
header before serving requests. The service exposes a small backend endpoint that
the frontend (or other callers such as GitHub Actions) can use to request code
suggestions from supported GPT models.
"""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


DEFAULT_MODEL = "gpt-4.1"
SUPPORTED_MODELS = {DEFAULT_MODEL, "gpt-5-codex"}


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gateway_token: str | None = Field(default=None, alias="GATEWAY_TOKEN")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "populate_by_name": True,
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings so configuration is evaluated once per process."""

    return Settings()


_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """Create (and cache) an OpenAI client once the API key is available."""

    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured.",
        )

    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client

app = FastAPI(title="GPT-4.1 Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your deployment requirements.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodePrompt(BaseModel):
    """Request payload sent by the frontend."""

    prompt: str
    model: str | None = None


class CodeResponse(BaseModel):
    """Response payload returned to the frontend."""

    result: str


@app.post("/api/codex", response_model=CodeResponse)
async def generate_code(payload: CodePrompt, request: Request) -> CodeResponse:
    """Call the configured GPT model to generate code for the provided prompt."""

    settings = get_settings()

    if settings.gateway_token:
        expected = f"Bearer {settings.gateway_token}"
        received = request.headers.get("Authorization")
        if received != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing bearer token.",
            )

    model_name = payload.model or DEFAULT_MODEL
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{model_name}'. Supported models: {sorted(SUPPORTED_MODELS)}",
        )

    client = get_openai_client()

    try:
        response = client.responses.create(
            model=model_name,
            input=f"Escreva código: {payload.prompt}",
        )
    except Exception as exc:  # pragma: no cover - defensive error translation
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CodeResponse(result=response.output_text)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Simple health endpoint for monitoring."""

    return {"status": "ok"}


__all__ = ["app"]
