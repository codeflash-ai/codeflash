from __future__ import annotations

import signal
from argparse import Namespace
from unittest.mock import Mock, call

from codeflash.optimization import optimizer as optimizer_module


def test_run_with_args_skips_unavailable_optional_signals(monkeypatch) -> None:
    cleanup_stale_worktrees = Mock()
    optimizer_instance = Mock(current_worktree=None)
    optimizer_class = Mock(return_value=optimizer_instance)
    getsignal = Mock(return_value="original-handler")
    signal_calls: list[tuple[object, object]] = []

    def fake_signal(signum: object, handler: object) -> None:
        signal_calls.append((signum, handler))

    monkeypatch.setattr(optimizer_module, "cleanup_stale_worktrees", cleanup_stale_worktrees)
    monkeypatch.setattr(optimizer_module, "Optimizer", optimizer_class)
    monkeypatch.setattr(signal, "getsignal", getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)
    monkeypatch.delattr(signal, "SIGHUP", raising=False)
    monkeypatch.delattr(signal, "SIGQUIT", raising=False)
    monkeypatch.delattr(signal, "SIGPIPE", raising=False)

    optimizer_module.run_with_args(Namespace())

    cleanup_stale_worktrees.assert_called_once_with()
    optimizer_class.assert_called_once()
    optimizer_instance.run.assert_called_once_with()
    assert getsignal.call_args_list == [call(signal.SIGTERM)]
    assert [signum for signum, _ in signal_calls] == [signal.SIGTERM, signal.SIGTERM]
