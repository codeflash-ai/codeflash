from os import getenv

from attrs import define, evolve

from constants import DEFAULT_API_URL, DEFAULT_APP_URL


@define
class ApiClient():
    api_key_header_name: str = "API-Key"
    client_type_header_name: str = "client-type"
    client_type_header_value: str = "sdk-python"

    @staticmethod
    def get_console_url() -> str:
        console_url = getenv("CONSOLE_URL", DEFAULT_API_URL)
        if DEFAULT_API_URL == console_url:
            return DEFAULT_APP_URL

        return console_url

    def with_api_key(self, api_key: str) -> "ApiClient": # ---> here is the problem with circular dependency, this makes libcst thinks that ApiClient needs an import despite it's already in the same file.
        """Get a new client matching this one with a new API key"""
        return evolve(self, api_key=api_key)

