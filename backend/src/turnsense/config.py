"""Server/worker settings. Provider secrets are read here and never leave the backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_prefix="TURNSENSE_",
        extra="ignore",
    )

    data_dir: Path = BACKEND_ROOT / "data"
    checkpoint_dir: Path = BACKEND_ROOT / "checkpoints"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    min_percentile_samples: int = 20
    max_concurrent_sessions: int = 8
    # Labelled manifest for the replay scenario (research/data.py format); optional.
    replay_manifest: Path | None = None
    # Optional split file: replay only its speaker-disjoint test_unseen recordings.
    replay_split: Path | None = None

    # --- LiveKit (transport) -------------------------------------------------
    livekit_url: str | None = Field(
        None, validation_alias=AliasChoices("LIVEKIT_URL", "TURNSENSE_LIVEKIT_URL")
    )
    livekit_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("LIVEKIT_API_KEY")
    )
    livekit_api_secret: SecretStr | None = Field(
        None, validation_alias=AliasChoices("LIVEKIT_API_SECRET")
    )
    token_ttl_s: int = Field(600, ge=60, le=3600)

    # --- Live pipeline providers ----------------------------------------------
    runtime_profile: Literal["hosted", "local"] = "hosted"
    live_vad: Literal["silero", "energy"] = "silero"
    deepgram_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("DEEPGRAM_API_KEY")
    )
    groq_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("GROQ_API_KEY"))
    stt_model: str = "nova-3"
    llm_model: str = "llama-3.1-8b-instant"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    tts_voice: str = "aura-2-thalia-en"
    tts_sample_rate: int = 24_000
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    whisper_model: str = "small.en"
    llm_max_tokens: int = 160
    system_prompt: str = (
        "You are a concise, friendly voice assistant. Answer in one to three short spoken "
        "sentences. Do not use lists, markdown or emoji."
    )
    # bounded waits so an adapter failure cannot deadlock a session (project.md §10)
    stt_final_wait_ms: int = 600
    llm_timeout_s: float = 15.0
    tts_timeout_s: float = 15.0

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)

    @property
    def trials_dir(self) -> Path:
        return self.data_dir / "trials"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
