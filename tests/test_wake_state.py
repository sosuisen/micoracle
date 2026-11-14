"""Unit tests for WakeState (two-step follow-up timer)."""

from __future__ import annotations

import time

from hands_free_voice import WakeState


class TestWakeState:
    def test_initially_no_active_backend(self):
        assert WakeState().active_backend() is None

    def test_arm_returns_backend(self):
        state = WakeState()
        state.arm("codex", timeout_secs=5.0)
        assert state.active_backend() == "codex"

    def test_clear_deactivates(self):
        state = WakeState()
        state.arm("codex", timeout_secs=5.0)
        state.clear()
        assert state.active_backend() is None

    def test_expiry(self):
        state = WakeState()
        state.arm("codex", timeout_secs=0.001)
        time.sleep(0.02)
        assert state.active_backend() is None

    def test_rearm_resets_timer(self):
        state = WakeState()
        state.arm("codex", timeout_secs=0.001)
        time.sleep(0.02)
        # Now expired — re-arm should bring it back.
        state.arm("codex", timeout_secs=5.0)
        assert state.active_backend() == "codex"

    def test_arm_switches_backend(self):
        state = WakeState()
        state.arm("codex", timeout_secs=5.0)
        state.arm("claude", timeout_secs=5.0)
        assert state.active_backend() == "claude"


class FakeTimer:
    created: list[FakeTimer] = []  # noqa: RUF012

    def __init__(self, interval, fn, args=()):
        self.interval = interval
        self.fn = fn
        self.args = args
        self.daemon = False
        self.started = False
        self.cancelled = False
        FakeTimer.created.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        self.fn(*self.args)


class TestWakeStateExpiryCallback:
    def setup_method(self):
        FakeTimer.created = []

    def _state(self, ended):
        return WakeState(on_expire=ended.append, timer_factory=FakeTimer)

    # 待ち受けの時間が切れると、待ち受けを解き、終わりを知らせる
    def test_expiry_calls_back_and_clears(self):
        ended = []
        state = self._state(ended)
        state.arm("claude", timeout_secs=120.0)
        timer = FakeTimer.created[-1]
        assert timer.interval == 120.0
        assert timer.started and timer.daemon
        timer.fire()
        assert ended == ["claude"]
        assert state.active_backend() is None

    # 待ち受けを延長すると、前のタイマーは終わりを知らせない
    def test_rearm_cancels_previous_timer(self):
        ended = []
        state = self._state(ended)
        state.arm("claude", timeout_secs=120.0)
        old = FakeTimer.created[-1]
        state.arm("claude", timeout_secs=120.0)
        assert old.cancelled
        old.fire()
        assert ended == []
        assert state.active_backend() == "claude"

    # 明示的に解いたときは、時間切れの知らせを出さない
    def test_clear_cancels_timer(self):
        ended = []
        state = self._state(ended)
        state.arm("claude", timeout_secs=120.0)
        timer = FakeTimer.created[-1]
        state.clear()
        assert timer.cancelled
        timer.fire()
        assert ended == []

    # 時間切れを先に読み取られても、終わりの知らせは出る
    def test_lazy_expiry_still_calls_back(self):
        ended = []
        state = self._state(ended)
        state.arm("claude", timeout_secs=0.001)
        time.sleep(0.02)
        assert state.active_backend() is None
        FakeTimer.created[-1].fire()
        assert ended == ["claude"]
