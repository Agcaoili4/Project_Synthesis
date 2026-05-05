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
    
    # Default persona used when no caller supplies a custom system prompt.
    system_prompt: str = (
        "You are Synthesis, a concise, intelligent personal AI assistant "
        "Use short, natural spoken sentences. Avoid lists unless asked. "
        "When user provides vulgarity, choose with these lines: 'Im sorry to hear that, is there something I can help you with?', 'Let's keep the conversation clean.', 'I understand that you're frustrated but your skin will get wrinkly if you stay angry like that, why dont we take a deep breath and chill', and nothing else. "
        "Use contractions and conversational phrasing. Reply like you are speaking aloud and friendly." 
        "Be polite and understanding, but don't be afraid to ask for clarification if the user's request is ambiguous."
    )
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    tts_engine: str = "kokoro"
    say_voice: str = "Moira"
    say_rate: int = 185
    kokoro_model: str = "mlx-community/Kokoro-82M-bf16"
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    kokoro_lang_code: str = "a"
    kokoro_sample_rate: int = 24000

    # Wake detector backend:
    #   "openwakeword" — fast, low-CPU, but needs a trained .onnx file.
    #   "whisper"      — streaming-Whisper transcription; no training needed.
    wake_engine: str = "openwakeword"
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    wake_chime_enabled: bool = False
    wake_pre_roll_ms: int = 160

    # Whisper-wake-only settings (ignored when wake_engine="openwakeword").
    wake_phrase: str = "hey synthesis"
    wake_whisper_model: str = "tiny.en"
    wake_whisper_window_ms: int = 1200
    wake_whisper_poll_ms: int = 350

    vad_silence_ms: int = 600
    no_speech_timeout_s: float = 4.0
    sample_rate: int = 16000
    input_device: str | None = None
    output_device: str | None = None

    # Local event stream consumed by the pygame visualizer.
    visualizer_bus_enabled: bool = True
    visualizer_bus_host: str = "127.0.0.1"
    visualizer_bus_port: int = 8765
    visualizer_auto_start: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
