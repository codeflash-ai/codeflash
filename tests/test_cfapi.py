from __future__ import annotations

from unittest.mock import Mock

from codeflash.api.cfapi import get_user_id


def test_get_user_id_suppresses_invalid_key_exit(monkeypatch) -> None:
    response = Mock()
    response.status_code = 403
    response.reason = "Forbidden"

    exit_with_message = Mock(side_effect=AssertionError("exit_with_message should not be called"))

    monkeypatch.setattr("codeflash.api.cfapi.ensure_codeflash_api_key", Mock(return_value=True))
    monkeypatch.setattr("codeflash.api.cfapi.make_cfapi_request", Mock(return_value=response))
    monkeypatch.setattr("codeflash.api.cfapi.exit_with_message", exit_with_message)

    get_user_id.cache_clear()
    try:
        assert get_user_id(suppress_errors=True) is None
    finally:
        get_user_id.cache_clear()

    exit_with_message.assert_not_called()
