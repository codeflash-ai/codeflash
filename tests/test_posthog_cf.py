from __future__ import annotations

from unittest.mock import Mock

from codeflash.telemetry import posthog_cf


def test_initialize_posthog_ignores_user_lookup_failures(monkeypatch) -> None:
    fake_client = Mock()
    fake_client.log = Mock()

    monkeypatch.setattr("posthog.Posthog", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr("codeflash.api.cfapi.get_user_id", Mock(side_effect=SystemExit(1)))
    monkeypatch.setattr(posthog_cf, "_posthog", None)

    posthog_cf.initialize_posthog(enabled=True)

    fake_client.capture.assert_not_called()


def test_ph_ignores_capture_failures(monkeypatch) -> None:
    fake_client = Mock()
    fake_client.capture.side_effect = RuntimeError("capture failed")

    monkeypatch.setattr(posthog_cf, "_posthog", fake_client)
    monkeypatch.setattr("codeflash.api.cfapi.get_user_id", Mock(return_value="user-123"))

    posthog_cf.ph("cli-test-event")

    fake_client.capture.assert_called_once()


def test_ph_uses_silent_user_lookup(monkeypatch) -> None:
    fake_client = Mock()
    get_user_id = Mock(return_value=None)

    monkeypatch.setattr(posthog_cf, "_posthog", fake_client)
    monkeypatch.setattr("codeflash.api.cfapi.get_user_id", get_user_id)

    posthog_cf.ph("cli-test-event")

    get_user_id.assert_called_once_with(suppress_errors=True)
