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

    # 声の合図では、待ち受けの終わりに何も話さない
    def test_end_is_silent(self):
        tts = FakeTTS()
        cues.VoiceCues(tts, wake_phrase="ok").end()
        assert tts.spoken == []


class TestToneCues:
    # トーンの合図では、待ち受けでラ、送信でファ、終わりでドを鳴らす
    def test_plays_la_fa_do(self):
        played = []
        c = cues.ToneCues(play=lambda freq: played.append(freq))
        c.wake()
        c.sent()
        c.end()
        assert played == [cues.LA_HZ, cues.FA_HZ, cues.DO_HZ]

    # 3つの音は、だんだん低くなる
    def test_tones_descend(self):
        assert cues.LA_HZ > cues.FA_HZ > cues.DO_HZ


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
