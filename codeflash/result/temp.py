from abc import abstractmethod
from typing import Any, List, Mapping, MutableMapping, Optional, Tuple

import requests
from requests.auth import AuthBase


class AbstractOauth2Authenticator(AuthBase):
    def __init__(
            self,
            refresh_token_error_status_codes: Tuple[int, ...] = (),
            refresh_token_error_key: str = "",
            refresh_token_error_values: Tuple[str, ...] = (),
    ) -> None:
        """
        If all of refresh_token_error_status_codes, refresh_token_error_key, and refresh_token_error_values are set,
        then http errors with such params will be wrapped in AirbyteTracedException.
        """
        self._refresh_token_error_status_codes = refresh_token_error_status_codes
        self._refresh_token_error_key = refresh_token_error_key
        self._refresh_token_error_values = refresh_token_error_values

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Attach the HTTP headers required to authenticate on the HTTP request"""
        request.headers.update(self.get_auth_header())
        return request

    def build_refresh_request_body(self) -> Mapping[str, Any]:
        """
        Returns the request body to set on the refresh request

        Override to define additional parameters
        """
        payload: MutableMapping[str, Any] = {
            "grant_type": self.get_grant_type(),
            "client_id": self.get_client_id(),
            "client_secret": self.get_client_secret(),
            "refresh_token": self.get_refresh_token(),
        }

        if self.get_scopes():
            payload["scopes"] = self.get_scopes()

        if self.get_refresh_request_body():
            for key, val in self.get_refresh_request_body().items():
                # We defer to existing oauth constructs over custom configured fields
                if key not in payload:
                    payload[key] = val

        return payload

    @abstractmethod
    def get_grant_type(self) -> str:
        """Returns grant_type specified for requesting access_token"""

    @abstractmethod
    def get_client_id(self) -> str:
        """The client id to authenticate"""

    @abstractmethod
    def get_client_secret(self) -> str:
        """The client secret to authenticate"""

    @abstractmethod
    def get_refresh_token(self) -> Optional[str]:
        """The token used to refresh the access token when it expires"""

    @abstractmethod
    def get_scopes(self) -> List[str]:
        """List of requested scopes"""

    @abstractmethod
    def get_refresh_request_body(self) -> Mapping[str, Any]:
        """Returns the request body to set on the refresh request"""
