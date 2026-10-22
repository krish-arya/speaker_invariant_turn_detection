"""Builds the live provider stack for the configured runtime profile."""

from __future__ import annotations

from dataclasses import dataclass

from turnsense.adapters.base import LLMAdapter, STTAdapter, TTSAdapter
from turnsense.config import Settings


class ProviderConfigError(RuntimeError):
    pass


@dataclass
class LiveProviders:
    stt: STTAdapter
    llm: LLMAdapter
    tts: TTSAdapter


def build_live_providers(settings: Settings) -> LiveProviders:
    if settings.runtime_profile == "local":
        from turnsense.adapters.local import KokoroTTS, WhisperSTT
        from turnsense.adapters.openai_compat import OllamaLLM

        return LiveProviders(
            stt=WhisperSTT(settings.whisper_model),
            llm=OllamaLLM(settings.ollama_url, settings.ollama_model, settings.llm_max_tokens),
            tts=KokoroTTS(),
        )

    from turnsense.adapters.deepgram import DeepgramSTT, DeepgramTTS
    from turnsense.adapters.openai_compat import groq_llm

    missing = [
        name
        for name, value in (
            ("DEEPGRAM_API_KEY", settings.deepgram_api_key),
            ("GROQ_API_KEY", settings.groq_api_key),
        )
        if value is None
    ]
    if missing:
        raise ProviderConfigError(f"hosted profile needs {', '.join(missing)} in backend/.env")
    dg = settings.deepgram_api_key.get_secret_value()  # type: ignore[union-attr]
    return LiveProviders(
        stt=DeepgramSTT(dg, settings.stt_model),
        llm=groq_llm(
            settings.groq_api_key.get_secret_value(),  # type: ignore[union-attr]
            settings.llm_model,
            settings.groq_base_url,
            settings.llm_max_tokens,
        ),
        tts=DeepgramTTS(dg, settings.tts_voice, settings.tts_sample_rate),
    )
