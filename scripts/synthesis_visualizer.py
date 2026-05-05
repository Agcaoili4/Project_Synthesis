"""
Synthesis voice visualizer — Siri-style reactive waveform.

Two modes, picked automatically:

    1. **Connected to the daemon's event bus** (default, when daemon is running)
         A background thread connects to 127.0.0.1:8765 and reads NDJSON
         events. The daemon publishes:
             - state transitions (idle / listening / transcribing /
               thinking / speaking) — these change the wave's palette and
               energy.
             - audio level frames tagged source="mic" (during LISTENING)
               or source="tts" (during SPEAKING) — these drive the shape
               of the wave.
         When connected, local mic capture is disabled — the daemon already
         owns the mic, and the bus tells us what to draw.

    2. **Standalone** (when --no-bus is passed or the daemon isn't running)
         Falls back to local mic capture so the script is still useful as
         a demo on its own.

Audio → animation pipeline:
    1. Whichever source is active (bus or local mic) produces level features:
       RMS amplitude + 3-band FFT energies, all in 0..1.
    2. Features are exponentially smoothed (one-pole lerp) to kill jitter.
    3. A row of horizontal control points each has position + velocity. We
       compute a target shape from the smoothed bands plus a touch of noise,
       then apply a spring force toward that target with damping. That's
       what makes the idle/listening wave feel elastic instead of glitchy.
    4. The smoothed control points are turned into a curve via Catmull-Rom
       interpolation, then rendered as multiple stacked semi-transparent
       polylines (decreasing alpha, increasing thickness) for a glow effect.
       Three layers with small phase delays give depth.
    5. During SPEAKING, the renderer morphs the wavelength into a reactive
       DNA-like double helix, using the violet, blue, and amber brand palette.
       When speech ends, it eases back into the waveform instead of snapping.
    6. Palette and energy targets shift smoothly with daemon state so the
       UI feels alive even when no audio is flowing (e.g., THINKING).

Run:
    .venv/bin/python scripts/synthesis_visualizer.py
    .venv/bin/python scripts/synthesis_visualizer.py --no-bus    # standalone
    .venv/bin/python scripts/synthesis_visualizer.py --no-mic    # bus only

Keys:
    SPACE — toggle local mic (only meaningful when not connected to bus)
    ESC / window close — quit
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame
import sounddevice as sd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 1280, 720
FPS = 60

SAMPLE_RATE = 44_100
CHUNK = 1024
RING_SECONDS = 1.0

AMP_SMOOTH = 0.18
BAND_SMOOTH = 0.22
COLOR_SMOOTH = 0.06       # slower than audio smoothing; makes palette shifts visible
MODE_TRANSITION_SMOOTH = 0.055

SPRING_STIFFNESS = 0.22
SPRING_DAMPING = 0.78

NUM_POINTS = 96
NUM_LAYERS = 3
GLOW_PASSES = 6

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BG_COLOR = (18, 13, 26)
PANEL_COLOR = (28, 20, 39)
TEXT_COLOR = (188, 169, 204)
AMBER = (242, 166, 95)
VIOLET = (143, 104, 223)
LAVENDER = (184, 126, 222)
BLUE = (117, 167, 255)

IDLE_AMPLITUDE = 0.06
IDLE_FREQ = 0.6
NOISE_STRENGTH = 0.04

DEFAULT_BUS_HOST = "127.0.0.1"
DEFAULT_BUS_PORT = 8765

# If bus levels stop arriving, keep animating from state instead of freezing.
LEVEL_FRESHNESS_S = 0.6


# Per-state palette and baseline motion.
STATE_RECIPES = {
    "idle":         {"left": (92, 70, 158),    "right": (143, 104, 223), "amp": 1.0, "label": "idle"},
    "listening":    {"left": (117, 167, 255),  "right": (184, 126, 222), "amp": 1.35, "label": "listening"},
    "transcribing": {"left": (143, 104, 223),  "right": (242, 166, 95),  "amp": 1.18, "label": "transcribing"},
    "thinking":     {"left": (184, 126, 222),  "right": (95, 76, 166),   "amp": 1.5, "label": "thinking"},
    "speaking":     {"left": (117, 167, 255),  "right": (242, 166, 95),  "amp": 1.85, "label": "speaking"},
}


# ---------------------------------------------------------------------------
# Bus client (subscribes to daemon NDJSON events on a background thread)
# ---------------------------------------------------------------------------

@dataclass
class SharedBusState:
    """Latest daemon event data guarded by a lock."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    connected: bool = False
    state: str = "idle"
    rms: float = 0.0
    low: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    last_level_at: float = 0.0
    level_source: str = ""   # "mic" / "tts" / ""

    def snapshot(self) -> "BusSnapshot":
        with self.lock:
            return BusSnapshot(
                connected=self.connected,
                state=self.state,
                rms=self.rms,
                low=self.low,
                mid=self.mid,
                high=self.high,
                last_level_at=self.last_level_at,
                level_source=self.level_source,
            )


@dataclass(frozen=True)
class BusSnapshot:
    connected: bool
    state: str
    rms: float
    low: float
    mid: float
    high: float
    last_level_at: float
    level_source: str


class BusClient(threading.Thread):
    """Connects to the daemon, reads NDJSON forever, auto-reconnects."""

    def __init__(self, host: str, port: int, shared: SharedBusState):
        super().__init__(daemon=True, name="visualizer-bus-client")
        self._host = host
        self._port = port
        self._shared = shared
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        backoff = 0.5
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self._host, self._port), timeout=2.0
                ) as sock:
                    sock.settimeout(None)
                    with self._shared.lock:
                        self._shared.connected = True
                    backoff = 0.5
                    self._read_forever(sock)
            except (ConnectionRefusedError, socket.timeout, OSError):
                # The daemon may start after the visualizer. Keep retrying quietly.
                pass
            finally:
                with self._shared.lock:
                    self._shared.connected = False
                    self._shared.state = "idle"
                    self._shared.last_level_at = 0.0

            self._stop_event.wait(backoff)
            backoff = min(backoff * 1.6, 4.0)

    def _read_forever(self, sock: socket.socket) -> None:
        # Read newline-delimited JSON without socket.makefile(); it handles
        # timeouts inconsistently across platforms.
        buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                data = sock.recv(4096)
            except OSError:
                return
            if not data:
                return
            buffer.extend(data)
            while True:
                nl = buffer.find(b"\n")
                if nl < 0:
                    break
                line = bytes(buffer[:nl])
                del buffer[:nl + 1]
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._apply(event)

    def _apply(self, event: dict) -> None:
        et = event.get("type")
        if et == "state":
            with self._shared.lock:
                self._shared.state = str(event.get("value", "idle"))
        elif et == "level":
            with self._shared.lock:
                self._shared.rms = float(event.get("rms", 0.0))
                self._shared.low = float(event.get("low", 0.0))
                self._shared.mid = float(event.get("mid", 0.0))
                self._shared.high = float(event.get("high", 0.0))
                self._shared.last_level_at = time.monotonic()
                self._shared.level_source = str(event.get("source", ""))
        # "hello" only confirms protocol version.


# ---------------------------------------------------------------------------
# Local mic ring (fallback path when there's no daemon)
# ---------------------------------------------------------------------------

class AudioRing:
    """Thread-safe rolling buffer of the most recent mono samples."""

    def __init__(self, sample_rate: int, seconds: float):
        self.size = int(sample_rate * seconds)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.write = 0
        self.lock = threading.Lock()

    def push(self, block: np.ndarray) -> None:
        if block.ndim > 1:
            block = block[:, 0]
        n = block.shape[0]
        with self.lock:
            end = self.write + n
            if end <= self.size:
                self.buf[self.write:end] = block
            else:
                first = self.size - self.write
                self.buf[self.write:] = block[:first]
                self.buf[:n - first] = block[first:]
            self.write = end % self.size

    def latest(self, n: int) -> np.ndarray:
        with self.lock:
            if n >= self.size:
                return self.buf.copy()
            start = (self.write - n) % self.size
            if start + n <= self.size:
                return self.buf[start:start + n].copy()
            return np.concatenate((self.buf[start:], self.buf[:n - (self.size - start)]))


def make_audio_stream(ring: AudioRing) -> sd.InputStream:
    def callback(indata, frames, time_info, status):  # noqa: ANN001
        ring.push(indata.copy())

    return sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK,
        callback=callback,
    )


# ---------------------------------------------------------------------------
# Audio analysis (used only in standalone/local-mic mode)
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatures:
    rms: float
    low: float
    mid: float
    high: float


_HANN = np.hanning(CHUNK).astype(np.float32)


def analyze(samples: np.ndarray) -> AudioFeatures:
    if samples.size < CHUNK:
        return AudioFeatures(0.0, 0.0, 0.0, 0.0)

    rms_raw = float(np.sqrt(np.mean(samples * samples)))
    rms = math.tanh(rms_raw * 6.0)

    spec = np.abs(np.fft.rfft(samples * _HANN))
    freqs = np.fft.rfftfreq(CHUNK, d=1.0 / SAMPLE_RATE)

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        if not np.any(m):
            return 0.0
        return float(spec[m].mean())

    low = math.tanh(band(60, 250) / 4.0)
    mid = math.tanh(band(250, 2000) / 2.0)
    high = math.tanh(band(2000, 8000) / 1.0)
    return AudioFeatures(rms, low, mid, high)


def lerp(current: float, target: float, alpha: float) -> float:
    return current + (target - current) * alpha


def lerp_color(c0: tuple, c1: tuple, alpha: float) -> tuple:
    return tuple(c0[i] + (c1[i] - c0[i]) * alpha for i in range(3))


# ---------------------------------------------------------------------------
# Wave model: spring-damped control points
# ---------------------------------------------------------------------------

class Wave:
    def __init__(self, n: int, phase_offset: float = 0.0):
        self.n = n
        self.x = np.linspace(0, 1, n, dtype=np.float32)
        self.y = np.zeros(n, dtype=np.float32)
        self.vy = np.zeros(n, dtype=np.float32)
        self.phase_offset = phase_offset

    def update(
        self,
        t: float,
        feats: AudioFeatures,
        active: bool,
        amp_scale: float,
        rng: np.random.Generator,
    ) -> None:
        idle_env = IDLE_AMPLITUDE * amp_scale * (
            0.6 + 0.4 * math.sin(2 * math.pi * IDLE_FREQ * t)
        )

        if active:
            amp = max(idle_env, feats.rms * 0.55 * amp_scale)
            phase = t * 1.6 + self.phase_offset
            target = (
                feats.low * np.sin(2 * np.pi * (1.0 * self.x) + phase)
                + feats.mid * np.sin(2 * np.pi * (2.5 * self.x) + phase * 1.3)
                + feats.high * np.sin(2 * np.pi * (5.0 * self.x) + phase * 1.7)
            )
            denom = max(1e-6, feats.low + feats.mid + feats.high)
            target = (target / denom) * amp
            target = target + rng.standard_normal(self.n).astype(np.float32) * NOISE_STRENGTH * amp
        else:
            phase = t * 1.2 + self.phase_offset
            # Idle / no-audio motion: a slow base sine plus a much slower
            # second harmonic. Together they look like breathing instead
            # of a single mechanical sine.
            target = idle_env * (
                np.sin(2 * np.pi * 1.0 * self.x + phase)
                + 0.4 * np.sin(2 * np.pi * 0.4 * self.x + phase * 0.6)
            )

        taper = np.sin(np.pi * self.x) ** 2
        target = target * taper

        accel = SPRING_STIFFNESS * (target - self.y) - (1.0 - SPRING_DAMPING) * self.vy
        self.vy += accel
        self.y += self.vy


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def catmull_rom(points: np.ndarray, samples_per_segment: int = 6) -> np.ndarray:
    p = points
    n = len(p)
    if n < 4:
        return p
    out = []
    ts = np.linspace(0, 1, samples_per_segment, endpoint=False)
    for i in range(1, n - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for tau in ts:
            tau2 = tau * tau
            tau3 = tau2 * tau
            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * tau
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tau2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * tau3
            )
            out.append(point)
    out.append(p[-2])
    return np.array(out, dtype=np.float32)


def color_at(left: tuple, right: tuple, u: float) -> tuple[int, int, int]:
    r = left[0] * (1 - u) + right[0] * u
    g = left[1] * (1 - u) + right[1] * u
    b = left[2] * (1 - u) + right[2] * u
    return int(r), int(g), int(b)


def draw_glow_curve(
    surface: pygame.Surface,
    curve: np.ndarray,
    base_alpha: int,
    palette: tuple[tuple, tuple],
) -> None:
    if len(curve) < 2 or base_alpha <= 0:
        return
    left, right = palette
    # Pygame expects plain Python ints here, not numpy scalar values.
    pts = [(int(round(float(p[0]))), int(round(float(p[1])))) for p in curve]
    n = len(pts)
    for pass_i in range(GLOW_PASSES, 0, -1):
        thickness = pass_i * 3
        alpha = int(base_alpha * (0.18 if pass_i > 1 else 1.0) / pass_i)
        if alpha <= 0:
            continue
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for i in range(n - 1):
            u = i / (n - 1)
            r, g, b = color_at(left, right, u)
            pygame.draw.line(layer, (r, g, b, alpha), pts[i], pts[i + 1], thickness)
        surface.blit(layer, (0, 0))


def draw_points_curve(
    surface: pygame.Surface,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    base_alpha: int,
    glow_passes: int = 5,
) -> None:
    if len(points) < 2 or base_alpha <= 0:
        return
    for pass_i in range(glow_passes, 0, -1):
        width = pass_i * 3
        alpha = int(base_alpha * (0.2 if pass_i > 1 else 1.0) / pass_i)
        if alpha <= 0:
            continue
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(layer, (*color, alpha), False, points, width)
        surface.blit(layer, (0, 0))


def draw_center_glow(surface: pygame.Surface, intensity: float, color: tuple) -> None:
    w, h = surface.get_size()
    glow = pygame.Surface((w, h), pygame.SRCALPHA)
    cx, cy = w // 2, h // 2
    base_alpha = int(20 + 60 * intensity)
    r0, g0, b0 = (int(color[0] * 0.55), int(color[1] * 0.55), int(color[2] * 0.95))
    for radius, a_mul in ((420, 0.25), (280, 0.45), (160, 0.8)):
        glow_color = (r0, g0, b0, int(base_alpha * a_mul))
        pygame.draw.circle(glow, glow_color, (cx, cy), radius)
    surface.blit(glow, (0, 0))


def draw_wave_layer(
    surface: pygame.Surface,
    waves: list[Wave],
    center: tuple[int, int],
    span: int,
    height: float,
    palette: tuple[tuple, tuple],
    alpha_scale: float = 1.0,
    layer_offset: int = 0,
) -> None:
    if alpha_scale <= 0.001:
        return
    cx, cy = center
    for layer_i, w in enumerate(waves):
        xs = (w.x - 0.5) * span + cx
        ys = cy - w.y * height
        pts = np.stack([xs, ys], axis=1)
        curve = catmull_rom(pts, samples_per_segment=8)
        alpha_index = layer_i + layer_offset
        base_alpha = [220, 150, 90][alpha_index] if alpha_index < 3 else 80
        draw_glow_curve(surface, curve, int(base_alpha * alpha_scale), palette)


def dna_points(
    size: tuple[int, int],
    t: float,
    feats: AudioFeatures,
    intensity: float,
    samples: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[float]]:
    w, h = size
    cx, cy = w / 2, h / 2
    height = h * 0.66
    top = cy - height / 2
    amplitude = 82 + 62 * min(1.0, intensity + feats.rms)
    phase = t * (2.2 + 1.5 * feats.high)
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    depths: list[float] = []

    for i in range(samples):
        u = i / (samples - 1)
        y = top + u * height
        twist = math.sin(u * math.tau * 2.4 + phase)
        pinch = 0.72 + 0.28 * math.sin(u * math.pi)
        x_offset = twist * amplitude * pinch
        left.append((cx - x_offset, y))
        right.append((cx + x_offset, y))
        depths.append((twist + 1.0) / 2.0)

    return left, right, depths


def draw_wave_to_dna_morph(
    surface: pygame.Surface,
    wave: Wave,
    center: tuple[int, int],
    span: int,
    wave_height: float,
    t: float,
    feats: AudioFeatures,
    palette: tuple[tuple, tuple],
    intensity: float,
    mix: float,
    alpha_scale: float = 1.0,
) -> None:
    """Render the front waveform as it morphs into a DNA helix.

    `mix` controls shape: 0 = both strands sit on the wave, 1 = full helix.
    `alpha_scale` controls visibility without changing the shape, which keeps
    the pure-wave and pure-DNA endpoints continuous.
    """
    if alpha_scale <= 0.005:
        return
    mix = smoothstep(mix)
    split = smoothstep(min(1.0, mix / 0.45))
    helix = smoothstep(max(0.0, (mix - 0.18) / 0.82))
    cx, cy = center
    samples = wave.n
    left_target, right_target, depths = dna_points(
        surface.get_size(), t, feats, intensity, samples
    )
    strand_left: list[tuple[int, int]] = []
    strand_right: list[tuple[int, int]] = []

    for i in range(samples):
        u = i / (samples - 1)
        wave_x = (float(wave.x[i]) - 0.5) * span + cx
        wave_y = cy - float(wave.y[i]) * wave_height * (1.0 - 0.18 * mix)
        taper = math.sin(math.pi * u) ** 0.8
        split_offset = 18.0 * split * taper

        start_left = (wave_x, wave_y - split_offset)
        start_right = (wave_x, wave_y + split_offset)
        lx = lerp(start_left[0], left_target[i][0], helix)
        ly = lerp(start_left[1], left_target[i][1], helix)
        rx = lerp(start_right[0], right_target[i][0], helix)
        ry = lerp(start_right[1], right_target[i][1], helix)
        strand_left.append((int(lx), int(ly)))
        strand_right.append((int(rx), int(ry)))

    left_color, right_color = palette
    base_alpha = int(round(210 * alpha_scale))
    draw_points_curve(surface, strand_left, color_at(left_color, right_color, 0.18), base_alpha)
    draw_points_curve(surface, strand_right, color_at(left_color, right_color, 0.82), base_alpha)

    # Add rungs after the strand split is visible; otherwise early transition
    # frames look noisy instead of intentional.
    rung_alpha_scale = smoothstep(max(0.0, (mix - 0.42) / 0.58)) * alpha_scale
    if rung_alpha_scale <= 0.005:
        return
    for i in range(0, samples, 7):
        color = color_at(BLUE, AMBER, depths[i])
        alpha = int((55 + 115 * depths[i]) * rung_alpha_scale)
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.line(layer, (*color, alpha), strand_left[i], strand_right[i], 3)
        pygame.draw.circle(layer, (*color, min(255, alpha + 28)), strand_left[i], 5)
        pygame.draw.circle(layer, (*color, min(255, alpha + 28)), strand_right[i], 5)
        surface.blit(layer, (0, 0))


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Synthesis voice visualizer")
    ap.add_argument("--bus-host", default=DEFAULT_BUS_HOST)
    ap.add_argument("--bus-port", type=int, default=DEFAULT_BUS_PORT)
    ap.add_argument("--no-bus", action="store_true",
                    help="Don't try to connect to the daemon event bus.")
    ap.add_argument("--no-mic", action="store_true",
                    help="Don't open the local microphone (use only bus events).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    pygame.init()
    pygame.display.set_caption("Synthesis — voice visualizer")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Helvetica", 14)

    shared = SharedBusState()
    bus_client: BusClient | None = None
    if not args.no_bus:
        bus_client = BusClient(args.bus_host, args.bus_port, shared)
        bus_client.start()

    ring: AudioRing | None = None
    stream: sd.InputStream | None = None
    audio_error: str | None = None
    if not args.no_mic:
        ring = AudioRing(SAMPLE_RATE, RING_SECONDS)
        try:
            stream = make_audio_stream(ring)
            stream.start()
        except Exception as exc:  # noqa: BLE001
            audio_error = f"mic unavailable: {exc}"
            ring = None

    waves = [Wave(NUM_POINTS, phase_offset=i * 0.6) for i in range(NUM_LAYERS)]
    rng = np.random.default_rng(1234)

    smooth = AudioFeatures(0.0, 0.0, 0.0, 0.0)
    smooth_amp_scale = 1.0
    smooth_left = STATE_RECIPES["idle"]["left"]
    smooth_right = STATE_RECIPES["idle"]["right"]
    dna_mix = 0.0
    mic_on = stream is not None

    t0 = pygame.time.get_ticks() / 1000.0
    running = True
    while running:
        clock.tick(FPS)
        t = pygame.time.get_ticks() / 1000.0 - t0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and stream is not None:
                    mic_on = not mic_on

        snap = shared.snapshot()
        recipe = STATE_RECIPES.get(snap.state, STATE_RECIPES["idle"])

        # Prefer daemon levels; standalone mode falls back to the local mic.
        bus_levels_fresh = (
            snap.connected
            and snap.last_level_at > 0
            and (time.monotonic() - snap.last_level_at) < LEVEL_FRESHNESS_S
        )

        if bus_levels_fresh:
            raw = AudioFeatures(rms=snap.rms, low=snap.low, mid=snap.mid, high=snap.high)
            source_label = f"bus·{snap.level_source or '?'}"
        elif ring is not None and mic_on and not snap.connected:
            samples = ring.latest(CHUNK)
            raw = analyze(samples)
            source_label = "mic"
        else:
            # Thinking and macOS `say` may have no level frames, so state alone
            # drives a gentle procedural motion.
            raw = AudioFeatures(0.0, 0.0, 0.0, 0.0)
            source_label = "procedural" if snap.connected else "silent"

        smooth = AudioFeatures(
            rms=lerp(smooth.rms, raw.rms, AMP_SMOOTH),
            low=lerp(smooth.low, raw.low, BAND_SMOOTH),
            mid=lerp(smooth.mid, raw.mid, BAND_SMOOTH),
            high=lerp(smooth.high, raw.high, BAND_SMOOTH),
        )

        smooth_amp_scale = lerp(smooth_amp_scale, recipe["amp"], COLOR_SMOOTH)
        smooth_left = lerp_color(smooth_left, recipe["left"], COLOR_SMOOTH)
        smooth_right = lerp_color(smooth_right, recipe["right"], COLOR_SMOOTH)
        dna_mix = lerp(
            dna_mix,
            1.0 if snap.state == "speaking" else 0.0,
            MODE_TRANSITION_SMOOTH,
        )

        active = smooth.rms > 0.04

        for w in waves:
            w.update(t, smooth, active, smooth_amp_scale, rng)

        screen.fill(BG_COLOR)
        draw_center_glow(screen, smooth.rms + 0.15 * (smooth_amp_scale - 1.0), smooth_right)

        cx, cy = WIDTH // 2, HEIGHT // 2
        wave_span = int(WIDTH * 0.78)
        wave_height = HEIGHT * 0.30

        # 0 = pure wave, 1 = DNA. Smoothstep removes edge pops at both ends.
        dna_alpha = smoothstep(dna_mix)
        palette = (smooth_left, smooth_right)
        dna_intensity = smooth.rms + 0.2 * smooth_amp_scale

        back_alpha = 1.0 - 0.55 * dna_alpha
        # The front wave is the strand source, so it fades as the morph appears.
        front_alpha = 1.0 - dna_alpha

        draw_wave_layer(
            screen,
            waves[1:],
            (cx, cy),
            wave_span,
            wave_height * (1.0 - 0.18 * dna_alpha),
            palette,
            alpha_scale=back_alpha,
            layer_offset=1,
        )
        if front_alpha > 0.01:
            draw_wave_layer(
                screen,
                [waves[0]],
                (cx, cy),
                wave_span,
                wave_height,
                palette,
                alpha_scale=front_alpha,
            )
        if dna_alpha > 0.01:
            draw_wave_to_dna_morph(
                screen,
                waves[0],
                (cx, cy),
                wave_span,
                wave_height,
                t,
                smooth,
                palette,
                dna_intensity,
                dna_mix,
                alpha_scale=dna_alpha,
            )

        link = "bus✓" if snap.connected else ("bus…" if bus_client else "bus off")
        if audio_error and not snap.connected:
            link = f"{link}  {audio_error}"
        hud = (
            f"Synthesis  ·  {recipe['label'].upper()}  ·  {link}  ·  src={source_label}  "
            f"·  rms={smooth.rms:.2f}  ·  SPACE mic  ESC quit"
        )
        screen.blit(font.render(hud, True, TEXT_COLOR), (16, HEIGHT - 26))

        pygame.display.flip()

    if bus_client is not None:
        bus_client.stop()
    if stream is not None:
        stream.stop()
        stream.close()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
