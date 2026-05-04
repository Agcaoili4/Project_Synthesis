from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    brain_host: str = "127.0.0.1"
    brain_port: int = 8000
    brain_api_token: str | None = None
    log_conversation_text: bool = False

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    llm_max_history: int = 20
    # The system prompt to make sure that the model always knows its identity and what it needed to do.
    system_prompt: str = (
        "You are Synthesis, a concise, intelligent personal AI assistant "
        "Use short, natural spoken sentences. Avoid lists unless asked. "
        "Use contractions and conversational phrasing. Reply like you are speaking aloud."

    )

    whisper_model: str = "small.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    tts_engine: str = "say"
    say_voice: str = "Moira"
    say_rate: int = 185

    # Wake up model
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5

    vad_silence_ms: int = 900
    sample_rate: int = 16000


@lru_cache
def get_settings() -> Settings:
    return Settings()
