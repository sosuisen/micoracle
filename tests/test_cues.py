"""Status cues: spoken words or short tones."""

from __future__ import annotations

import numpy as np
import pytest

import cues


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def speak(self, phrase):
        self.spoken.append(phrase)


class TestVoiceCues:
    # 声の合図では、待ち受けで確認の言葉を、送信で「sent」を話す
    def test_speaks_ack_and_sent(self):
        tts = FakeTTS()
        c = cues.VoiceCues(tts, wake_phrase="ok")
        c.wake()
        c.sent()
        assert tts.spoken == ["ok", "sent"]


class TestToneCues:
    # トーンの合図では、待ち受けでソ、送信で下のドを鳴らす
    def test_plays_sol_then_lower_do(self):
        played = []
        c = cues.ToneCues(play=lambda freq: played.append(freq))
        c.wake()
        c.sent()
        assert played == [cues.SOL_HZ, cues.DO_HZ]
        assert cues.DO_HZ < cues.SOL_HZ


class TestToneSamples:
    # トーンの波形は、指定した長さで、音量の上限を超えない
    def test_length_and_peak(self):
        samples = cues.tone_samples(440.0, duration_ms=100, sample_rate=16000)
        assert samples.dtype == np.float32
        assert len(samples) == 1600
        assert np.max(np.abs(samples)) <= cues.TONE_VOLUME

    # 波形の始めと終わりは0で、クリック音が出ない
    def test_fades_in_and_out(self):
        samples = cues.tone_samples(440.0, duration_ms=100, sample_rate=16000)
        assert samples[0] == pytest.approx(0.0)
        assert samples[-1] == pytest.approx(0.0, abs=1e-3)


class TestMakeCues:
    # 「tone」を選ぶと、トーンの合図になる
    def test_tone_style(self):
        assert isinstance(cues.make_cues("tone", FakeTTS(), "ok"), cues.ToneCues)

    # 既定は、声の合図である
    def test_default_is_voice(self):
        assert isinstance(cues.make_cues("", FakeTTS(), "ok"), cues.VoiceCues)

    # 知らない値は、起動時にエラーになる
    def test_unknown_style_raises(self):
        with pytest.raises(ValueError):
            cues.make_cues("beep", FakeTTS(), "ok")
