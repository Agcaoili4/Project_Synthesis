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

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    llm_max_history: int = 20
    system_prompt: str = (
        "You are Synthesis, a concise, intelligent personal AI assistant "
        "modeled after JARVIS / FRIDAY. Reply in 1-3 sentences unless the "
        "user asks for more detail. Speak naturally — your reply will be "
        "spoken aloud."
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
