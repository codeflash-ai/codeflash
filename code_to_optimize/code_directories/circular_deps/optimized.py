from __future__ import annotations

import urllib.parse
from os import getenv

from attrs import define
from api_client import ApiClient
from constants import DEFAULT_API_URL, DEFAULT_APP_URL


@define
class ApiClient():

    @staticmethod
    def get_console_url() -> str:
        # Cache env lookup for speed
        console_url = getenv("CONSOLE_URL")
        if not console_url or console_url == DEFAULT_API_URL:
            return DEFAULT_APP_URL
        return console_url

# Pre-parse netlocs that are checked frequently to avoid parsing repeatedly
_DEFAULT_APP_URL_NETLOC = urllib.parse.urlparse(DEFAULT_APP_URL).netloc
_DEFAULT_API_URL_NETLOC = urllib.parse.urlparse(DEFAULT_API_URL).netloc

def get_dest_url(url: str) -> str:
    destination = url if url else ApiClient.get_console_url()
    # Replace only if 'console.' is at the beginning to avoid partial matches
    if destination.startswith("console."):
        destination = "api." + destination[len("console."):]
    else:
        destination = destination.replace("console.", "api.", 1)

    parsed_url = urllib.parse.urlparse(destination)
    if parsed_url.netloc == _DEFAULT_APP_URL_NETLOC or parsed_url.netloc == _DEFAULT_API_URL_NETLOC:
        return f"{DEFAULT_APP_URL}api/traces"
    return f"{parsed_url.scheme}://{parsed_url.netloc}/traces"