from app.infrastructure.audio.io import resolve_audio_device


def test_resolve_audio_device_keeps_default_when_unset():
    assert resolve_audio_device(None) is None
    assert resolve_audio_device("") is None
    assert resolve_audio_device("   ") is None
    assert resolve_audio_device("# blank = system default") is None


def test_resolve_audio_device_parses_numeric_index():
    assert resolve_audio_device("2") == 2
    assert resolve_audio_device(3) == 3


def test_resolve_audio_device_keeps_device_name():
    assert resolve_audio_device("MacBook Pro Microphone") == "MacBook Pro Microphone"
