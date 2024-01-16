import time
import webbrowser
from typing import Optional

import click
import requests

from codeflash.version import __version__ as version

CF_BASE_URL = "https://app.codeflash.ai"
LOGIN_URL = f"{CF_BASE_URL}/login"  # Replace with your actual URL
POLLING_URL = f"{CF_BASE_URL}/api/get-token"  # Replace with your actual polling endpoint
POLLING_INTERVAL = 10  # Polling interval in seconds
MAX_POLLING_ATTEMPTS = 30  # Maximum number of polling attempts

CODEFLASH_LOGO: str = (
    "\n"
    r"              __    _____         __ " + "\n"
    r" _______  ___/ /__ / _/ /__ ____ / / " + "\n"
    r"/ __/ _ \/ _  / -_) _/ / _ `(_-</ _ \ " + "\n"
    r"\__/\___/\_,_/\__/_//_/\_,_/___/_//_/" + "\n"
    f"{('v'+version).rjust(46)}\n"
    "                          https://codeflash.ai\n"
    "\n"
)


def open_login_page(session_id: str) -> str:
    webbrowser.open(f"{LOGIN_URL}?session_id={session_id}")
    return "Login page opened in your browser. Please complete the login process."


def poll_for_token(session_id: str) -> Optional[str]:
    for _ in range(MAX_POLLING_ATTEMPTS):
        response = requests.get(POLLING_URL, params={"session_id": session_id})
        if response.status_code == 200 and response.json().get("token"):
            return response.json()["token"]
        time.sleep(POLLING_INTERVAL)
    return None


@click.command()
def login():
    """
    Login to GitHub and save the access token.
    """
    click.echo("Starting GitHub login process...")
    session_id = "unique_session_id"  # Generate or retrieve a unique session ID
    click.echo(open_login_page(session_id))
    token = poll_for_token(session_id)
    if token:
        save_token(token)
        click.echo("Successfully logged in and token saved.")
    else:
        click.echo("Login failed. Please try again.")


# Function to save the token locally
def save_token(token: str):
    with open("github_token.txt", "w") as file:
        file.write(token)
    click.echo("Token saved successfully.")


# Main function
@click.group()
def cli():
    pass


cli.add_command(login)

if __name__ == "__main__":
    cli()
