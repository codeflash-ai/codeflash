from os import getenv
from typing import Optional

from attrs import define, evolve, field

from code_to_optimize.code_directories.circular_deps.constants import DEFAULT_API_URL, DEFAULT_APP_URL


@define
class GalileoApiClient():
    """A Client which has been authenticated for use on secured endpoints

    The following are accepted as keyword arguments and will be used to construct httpx Clients internally:

        ``base_url``: The base URL for the API, all requests are made to a relative path to this URL
            This can also be set via the GALILEO_CONSOLE_URL environment variable

        ``api_key``: The API key to be sent with every request
            This can also be set via the GALILEO_API_KEY environment variable

        ``cookies``: A dictionary of cookies to be sent with every request

        ``headers``: A dictionary of headers to be sent with every request

        ``timeout``: The maximum amount of a time a request can take. API functions will raise
        httpx.TimeoutException if this is exceeded.

        ``verify_ssl``: Whether or not to verify the SSL certificate of the API server. This should be True in production,
        but can be set to False for testing purposes.

        ``follow_redirects``: Whether or not to follow redirects. Default value is False.

        ``httpx_args``: A dictionary of additional arguments to be passed to the ``httpx.Client`` and ``httpx.AsyncClient`` constructor.

    Attributes:
        raise_on_unexpected_status: Whether or not to raise an errors.UnexpectedStatus if the API returns a
            status code that was not documented in the source OpenAPI document. Can also be provided as a keyword
            argument to the constructor.
        token: The token to use for authentication
        prefix: The prefix to use for the Authorization header
        auth_header_name: The name of the Authorization header
    """

    _base_url: Optional[str] = field(factory=lambda: GalileoApiClient.get_api_url(), kw_only=True, alias="base_url")
    _api_key: Optional[str] = field(factory=lambda: getenv("GALILEO_API_KEY", None), kw_only=True, alias="api_key")
    token: Optional[str] = None

    api_key_header_name: str = "Galileo-API-Key"
    client_type_header_name: str = "client-type"
    client_type_header_value: str = "sdk-python"

    @staticmethod
    def get_console_url() -> str:
        console_url = getenv("GALILEO_CONSOLE_URL", DEFAULT_API_URL)
        if DEFAULT_API_URL == console_url:
            return DEFAULT_APP_URL

        return console_url

    def with_api_key(self, api_key: str) -> "GalileoApiClient":
        """Get a new client matching this one with a new API key"""
        if self._client is not None:
            self._client.headers.update({self.api_key_header_name: api_key})
        if self._async_client is not None:
            self._async_client.headers.update({self.api_key_header_name: api_key})
        return evolve(self, api_key=api_key)

    @staticmethod
    def get_api_url(base_url: Optional[str] = None) -> str:
        api_url = base_url or getenv("GALILEO_CONSOLE_URL", DEFAULT_API_URL)
        if api_url is None:
            raise ValueError("base_url or GALILEO_CONSOLE_URL must be set")
        if any(map(api_url.__contains__, ["localhost", "127.0.0.1"])):
            api_url = "http://localhost:8088"
        else:
            api_url = api_url.replace("app.galileo.ai", "api.galileo.ai").replace("console", "api")
        return api_url
