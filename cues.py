"""Status cues for the two-step wake flow: spoken words or short tones.

A cue tells the user three things: "I am listening" after a bare wake word,
"sent" after the command is pasted, and "I stopped listening".
Tones are quicker than words. They also rarely come back through the mic
as text.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

# The three tones step down: listening, sent, then listening ended.
LA_HZ = 1760.0          # A6: listening
FA_HZ = 1396.91         # F6: sent
DO_HZ = 1046.5          # C6: listening ended
TONE_MS = 250
TONE_FADE_MS = 10       # fade in and out. A hard edge clicks.
TONE_VOLUME = 0.3
TONE_SAMPLE_RATE = 44100


class Cues(Protocol):
    def wake(self) -> None: ...
    def sent(self) -> None: ...
    def end(self) -> None: ...


class _Speaker(Protocol):
    def speak(self, phrase: str) -> None: ...


class VoiceCues:
    """Speak short words through the TTS backend."""

    def __init__(self, tts: _Speaker, wake_phrase: str) -> None:
        self._tts = tts
        self._wake_phrase = wake_phrase

    def wake(self) -> None:
        self._tts.speak(self._wake_phrase)

    def sent(self) -> None:
        self._tts.speak("sent")

    def end(self) -> None:
        # The voice mode has always ended listening in silence.
        return


class ToneCues:
    """Play LA when listening, FA when sent, and DO when listening ends."""

    def __init__(self, play: Callable[[float], None] | None = None) -> None:
        self._play = play or play_tone

    def wake(self) -> None:
        self._play(LA_HZ)

    def sent(self) -> None:
        self._play(FA_HZ)

    def end(self) -> None:
        self._play(DO_HZ)


def tone_samples(
    freq: float, duration_ms: int = TONE_MS, sample_rate: int = TONE_SAMPLE_RATE,
) -> np.ndarray:
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n) / sample_rate
    wave = TONE_VOLUME * np.sin(2 * np.pi * freq * t)
    fade = min(int(sample_rate * TONE_FADE_MS / 1000), n // 2)
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[n - fade:] = np.linspace(1.0, 0.0, fade)
    return (wave * envelope).astype(np.float32)


def play_tone(freq: float) -> None:
    """Start the tone and return at once. A failed cue must not stop dictation."""
    try:
        import sounddevice as sd

        sd.play(tone_samples(freq), TONE_SAMPLE_RATE)
    except Exception as exc:  # noqa: BLE001
        print(f"[cue error] {exc}", flush=True)


def make_cues(style: str, tts: _Speaker, wake_phrase: str) -> Cues:
    style = (style or "voice").strip().lower()
    if style == "voice":
        return VoiceCues(tts, wake_phrase)
    if style == "tone":
        return ToneCues()
    raise ValueError(f"Unknown cue style: {style!r}. Valid: voice, tone")
