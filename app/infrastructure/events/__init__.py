"""Local-only event bus for UI surfaces (visualizer, future dashboards)."""

from app.infrastructure.events.visualizer_bus import (
    VisualizerEventBus,
    compute_audio_levels,
)

__all__ = ["VisualizerEventBus", "compute_audio_levels"]
