from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser

import click
import requests

from codeflash.api.cfapi import get_cfapi_base_urls


class OAuthHandler:
    """Handle OAuth PKCE flow for CodeFlash authentication."""

    def __init__(self) -> None:
        self.code: str | None = None
        self.state: str | None = None
        self.error: str | None = None
        self.theme: str | None = None
        self.is_complete = False
        self.token_error: str | None = None
        self.manual_code: str | None = None
        self.lock = threading.Lock()

    def create_callback_handler(self) -> type[http.server.BaseHTTPRequestHandler]:
        """Create HTTP handler for OAuth callback."""
        oauth_handler = self

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            server_version = "CFHTTP"

            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)

                if parsed.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    status = {
                        "success": oauth_handler.token_error is None and oauth_handler.code is not None,
                        "error": oauth_handler.token_error,
                    }
                    self.wfile.write(json.dumps(status).encode())
                    return

                if parsed.path != "/callback":
                    self.send_response(404)
                    self.end_headers()
                    return

                params = urllib.parse.parse_qs(parsed.query)

                with oauth_handler.lock:
                    if not oauth_handler.is_complete:
                        oauth_handler.code = params.get("code", [None])[0]
                        oauth_handler.state = params.get("state", [None])[0]
                        oauth_handler.error = params.get("error", [None])[0]
                        oauth_handler.theme = params.get("theme", ["light"])[0]
                        oauth_handler.is_complete = True

                # Send HTML response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                html_content = self._get_html_response()
                self.wfile.write(html_content.encode())

            def _get_html_response(self) -> str:
                """Return simple HTML response."""
                theme = oauth_handler.theme or "light"
                if oauth_handler.error:
                    return self._get_error_html(oauth_handler.error, theme)
                if oauth_handler.code:
                    return self._get_loading_html(theme)
                return self._get_error_html("unauthorized", theme)

            @staticmethod
            def _get_loading_html(theme: str = "light") -> str:
                """Return loading state while exchanging token."""
                theme_class = "dark" if theme == "dark" else ""
                return f"""
<!DOCTYPE html>
<html class="{theme_class}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeFlash Authentication</title>
    <style>
        :root {{
            --background: hsl(0, 0%, 99%);
            --foreground: hsl(222.2, 84%, 4.9%);
            --card: hsl(0, 0%, 100%);
            --card-foreground: hsl(222.2, 84%, 4.9%);
            --primary: hsl(38, 100%, 63%);
            --primary-foreground: hsl(0, 6%, 4%);
            --muted: hsl(41, 20%, 96%);
            --muted-foreground: hsl(41, 8%, 46%);
            --border: hsl(41, 30%, 90%);
            --destructive: hsl(0, 84.2%, 60.2%);
            --destructive-foreground: hsl(0, 0%, 100%);
            --radius: 0.5rem;
            --success: hsl(142, 76%, 36%);
            --success-foreground: hsl(0, 0%, 100%);
        }}

        html.dark {{
            --background: hsl(0, 6%, 5%);
            --foreground: hsl(0, 0%, 100%);
            --card: hsl(0, 3%, 11%);
            --card-foreground: hsl(0, 0%, 100%);
            --primary: hsl(38, 100%, 63%);
            --primary-foreground: hsl(222.2, 47.4%, 11.2%);
            --muted: hsl(48, 15%, 20%);
            --muted-foreground: hsl(48, 20%, 65%);
            --border: hsl(48, 20%, 25%);
            --destructive: hsl(0, 62.8%, 30.6%);
            --destructive-foreground: hsl(0, 0%, 100%);
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--background);
            color: var(--foreground);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
        }}

        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            background: linear-gradient(to bottom,
                hsl(38, 100%, 63%, 0.1),
                hsl(38, 100%, 63%, 0.05),
                transparent);
            pointer-events: none;
            z-index: 0;
        }}

        body::after {{
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(to right, rgba(128, 128, 128, 0.03) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(128, 128, 128, 0.03) 1px, transparent 1px);
            background-size: 24px 24px;
            pointer-events: none;
            z-index: 0;
        }}

        .container {{
            max-width: 420px;
            width: 100%;
            position: relative;
            z-index: 1;
        }}

        .logo-container {{
            display: flex;
            justify-content: center;
            margin-bottom: 48px;
        }}

        .logo {{
            height: 40px;
            width: auto;
        }}

        .logo-light {{
            display: block;
        }}

        .logo-dark {{
            display: none;
        }}

        html.dark .logo-light {{
            display: none;
        }}

        html.dark .logo-dark {{
            display: block;
        }}

        .card {{
            background: var(--card);
            color: var(--card-foreground);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            padding: 48px;
            animation: fadeIn 0.3s ease-out forwards;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .icon-container {{
            width: 48px;
            height: 48px;
            background: hsl(38, 100%, 63%, 0.1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px;
        }}

        .spinner {{
            width: 24px;
            height: 24px;
            border: 2px solid var(--border);
            border-top-color: var(--primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .success-icon {{
            width: 64px;
            height: 64px;
            background: hsl(142, 76%, 36%, 0.1);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px;
        }}

        .success-checkmark {{
            width: 32px;
            height: 32px;
            stroke: hsl(142, 76%, 36%);
        }}

        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin: 0 0 12px;
            color: var(--card-foreground);
            text-align: center;
        }}

        p {{
            color: var(--muted-foreground);
            margin: 0;
            font-size: 14px;
            line-height: 1.5;
            text-align: center;
        }}

        .error-box {{
            background: var(--destructive);
            color: var(--destructive-foreground);
            padding: 14px 18px;
            border-radius: 8px;
            margin-top: 24px;
            font-size: 14px;
            line-height: 1.5;
        }}

        @media (max-width: 480px) {{
            .card {{
                padding: 32px 24px;
            }}

            h1 {{
                font-size: 20px;
            }}

            .logo {{
                height: 32px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo-container">
            <img src="https://app.codeflash.ai/images/codeflash_light.svg" alt="CodeFlash" class="logo logo-light" />
            <img src="https://app.codeflash.ai/images/codeflash_darkmode.svg" alt="CodeFlash" class="logo logo-dark" />
        </div>
        <div class="card" id="content">
            <div class="icon-container">
                <div class="spinner"></div>
            </div>
            <h1>Authenticating</h1>
            <p>Please wait while we verify your credentials...</p>
        </div>
    </div>

    <script>
        let pollCount = 0;
        const maxPolls = 60;

        function checkStatus() {{
            fetch('/status')
                .then(res => res.json())
                .then(data => {{
                    if (data.success) {{
                        showSuccess();
                    }} else if (data.error) {{
                        showError(data.error);
                    }} else if (pollCount < maxPolls) {{
                        pollCount++;
                        setTimeout(checkStatus, 500);
                    }} else {{
                        showError('Authentication timed out. Please try again.');
                    }}
                }})
                .catch(() => {{
                    if (pollCount < maxPolls) {{
                        pollCount++;
                        setTimeout(checkStatus, 500);
                    }}
                }});
        }}

        function showSuccess() {{
            document.getElementById('content').innerHTML = `
                <div class="success-icon">
                    <svg class="success-checkmark" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                </div>
                <h1>Success!</h1>
                <p>Authentication completed. You can now close this window.</p>
            `;
        }}

        function showError(message) {{
            document.getElementById('content').innerHTML = `
                <div class="icon-container" style="background: hsl(0, 84.2%, 60.2%, 0.1);">
                    <svg width="24" height="24" fill="none" stroke="hsl(0, 84.2%, 60.2%)" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" stroke-width="2"></circle>
                        <line x1="12" y1="8" x2="12" y2="12" stroke-width="2" stroke-linecap="round"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16" stroke-width="2" stroke-linecap="round"></line>
                    </svg>
                </div>
                <h1>Authentication Failed</h1>
                <div class="error-box">${{message}}</div>
            `;
        }}

        setTimeout(checkStatus, 1000);
    </script>
</body>
</html>
                """

            @staticmethod
            def _get_error_html(error_message: str, theme: str = "light") -> str:
                """Return error state HTML."""
                theme_class = "dark" if theme == "dark" else ""
                return f"""
<!DOCTYPE html>
<html class="{theme_class}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeFlash Authentication</title>
    <style>
        :root {{{{
            --background: hsl(0, 0%, 99%);
            --foreground: hsl(222.2, 84%, 4.9%);
            --card: hsl(0, 0%, 100%);
            --card-foreground: hsl(222.2, 84%, 4.9%);
            --primary: hsl(38, 100%, 63%);
            --muted-foreground: hsl(41, 8%, 46%);
            --border: hsl(41, 30%, 90%);
            --destructive: hsl(0, 84.2%, 60.2%);
            --destructive-foreground: hsl(0, 0%, 100%);
            --radius: 0.5rem;
        }}}}

        html.dark {{{{
            --background: hsl(0, 6%, 5%);
            --foreground: hsl(0, 0%, 100%);
            --card: hsl(0, 3%, 11%);
            --card-foreground: hsl(0, 0%, 100%);
            --primary: hsl(38, 100%, 63%);
            --muted-foreground: hsl(48, 20%, 65%);
            --border: hsl(48, 20%, 25%);
            --destructive: hsl(0, 62.8%, 30.6%);
            --destructive-foreground: hsl(0, 0%, 100%);
        }}}}

        * {{{{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}}}

        body {{{{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--background);
            color: var(--foreground);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
        }}}}

        body::before {{{{
            content: '';
            position: fixed;
            inset: 0;
            background: linear-gradient(to bottom,
                hsl(38, 100%, 63%, 0.1),
                hsl(38, 100%, 63%, 0.05),
                transparent);
            pointer-events: none;
            z-index: 0;
        }}}}

        body::after {{{{
            content: '';
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(to right, rgba(128, 128, 128, 0.03) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(128, 128, 128, 0.03) 1px, transparent 1px);
            background-size: 24px 24px;
            pointer-events: none;
            z-index: 0;
        }}}}

        .container {{{{
            max-width: 420px;
            width: 100%;
            position: relative;
            z-index: 1;
        }}}}

        .logo-container {{{{
            display: flex;
            justify-content: center;
            margin-bottom: 48px;
        }}}}

        .logo {{{{
            height: 40px;
            width: auto;
        }}}}

        .logo-light {{{{
            display: block;
        }}}}

        .logo-dark {{{{
            display: none;
        }}}}

        html.dark .logo-light {{{{
            display: none;
        }}}}

        html.dark .logo-dark {{{{
            display: block;
        }}}}

        .card {{{{
            background: var(--card);
            color: var(--card-foreground);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            padding: 48px;
            animation: fadeIn 0.3s ease-out forwards;
        }}}}

        @keyframes fadeIn {{{{
            from {{{{
                opacity: 0;
                transform: translateY(10px);
            }}}}
            to {{{{
                opacity: 1;
                transform: translateY(0);
            }}}}
        }}}}

        .icon-container {{{{
            width: 80px;
            height: 80px;
            background: hsl(38, 100%, 50%, 0.1);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 24px;
        }}}}

        h1 {{{{
            font-size: 24px;
            font-weight: 600;
            margin: 0 0 12px;
            color: var(--card-foreground);
            text-align: center;
        }}}}

        .error-box {{{{
            background: var(--destructive);
            color: var(--destructive-foreground);
            padding: 14px 18px;
            border-radius: 8px;
            margin-top: 24px;
            font-size: 14px;
            line-height: 1.5;
            text-align: center;
        }}}}

        @media (max-width: 480px) {{{{
            .card {{{{
                padding: 32px 24px;
            }}}}

            h1 {{{{
                font-size: 20px;
            }}}}

            .logo {{{{
                height: 32px;
            }}}}
        }}}}
    </style>
</head>
<body>
    <div class="container">
        <div class="logo-container">
            <img src="https://app.codeflash.ai/images/codeflash_light.svg" alt="CodeFlash" class="logo logo-light" />
            <img src="https://app.codeflash.ai/images/codeflash_darkmode.svg" alt="CodeFlash" class="logo logo-dark" />
        </div>
        <div class="card">
            <div class="icon-container">
                <svg width="48" height="48" fill="none" stroke="hsl(38, 100%, 50%)" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10" stroke-width="2"></circle>
                    <line x1="12" y1="8" x2="12" y2="12" stroke-width="2" stroke-linecap="round"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16" stroke-width="2" stroke-linecap="round"></line>
                </svg>
            </div>
            <h1>Authentication Failed</h1>
            <div class="error-box">{error_message}</div>
        </div>
    </div>
</body>
</html>
                """

            def log_message(self, fmt: str, *args: object) -> None:
                """Suppress log messages."""

        return CallbackHandler

    @staticmethod
    def get_free_port() -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        code_verifier = "".join(
            secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~") for _ in range(64)
        )
        code_challenge = base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).rstrip(b"=").decode()
        return code_verifier, code_challenge

    def start_local_server(self, port: int) -> http.server.HTTPServer:
        """Start local HTTP server for OAuth callback."""
        handler_class = self.create_callback_handler()
        httpd = http.server.HTTPServer(("localhost", port), handler_class)

        def serve_forever_wrapper() -> None:
            httpd.serve_forever()

        server_thread = threading.Thread(target=serve_forever_wrapper)
        server_thread.daemon = True
        server_thread.start()

        return httpd

    def exchange_code_for_token(self, code: str, code_verifier: str, redirect_uri: str) -> str | None:
        """Exchange authorization code for API token."""
        token_url = f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "client_id": "cf-cli-app",
        }

        try:
            resp = requests.post(token_url, json=data, timeout=10)
            resp.raise_for_status()
            token_json = resp.json()
            api_key = token_json.get("access_token")

            if not api_key:
                self.token_error = "No access token in response"  # noqa: S105
                return None

        except requests.exceptions.HTTPError:
            self.token_error = "Unauthorized"  # noqa: S105
            return None
        else:
            return api_key


def get_browser_name_fallback() -> str | None:
    try:
        controller = webbrowser.get()
        # controller.name exists for most browser controllers
        return getattr(controller, "name", None)
    except Exception:
        return None


def should_attempt_browser_launch() -> bool:
    # A list of browser names that indicate we should not attempt to open a
    # web browser for the user.
    browser_blocklist = ["www-browser", "lynx", "links", "w3m", "elinks", "links2"]
    browser_env = os.environ.get("BROWSER") or get_browser_name_fallback()
    if browser_env and browser_env in browser_blocklist:
        return False

    # Common environment variables used in CI/CD or other non-interactive shells.
    if os.environ.get("CI") or os.environ.get("DEBIAN_FRONTEND") == "noninteractive":
        return False

    # The presence of SSH_CONNECTION indicates a remote session.
    # We should not attempt to launch a browser unless a display is explicitly available
    # (checked below for Linux).
    is_ssh = bool(os.environ.get("SSH_CONNECTION"))

    # On Linux, the presence of a display server is a strong indicator of a GUI.
    if sys.platform == "linux":
        # These are environment variables that can indicate a running compositor on
        # Linux.
        display_variables = ["DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"]
        has_display = any(os.environ.get(v) for v in display_variables)
        if not has_display:
            return False

    # If in an SSH session on a non-Linux OS (e.g., macOS), don't launch browser.
    # The Linux case is handled above (it's allowed if DISPLAY is set).
    if is_ssh and sys.platform != "linux":
        return False

    # For non-Linux OSes, we generally assume a GUI is available
    # unless other signals (like SSH) suggest otherwise.
    # The `open` command's error handling will catch final edge cases.
    return True


def _wait_for_manual_code_input(oauth: OAuthHandler) -> None:
    """Thread function to wait for manual code input."""
    try:
        code = input()
        with oauth.lock:
            if not oauth.is_complete:
                oauth.manual_code = code.strip()
                oauth.is_complete = True
    except Exception:  # noqa: S110
        pass


def perform_oauth_signin() -> str | None:
    """Perform OAuth PKCE flow and return API key if successful.

    Returns None if failed.
    """
    oauth = OAuthHandler()

    # Setup PKCE
    port = oauth.get_free_port()
    code_verifier, code_challenge = oauth.generate_pkce_pair()
    state = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(16))

    # Build authorization URLs for both local and remote
    local_redirect_uri = f"http://localhost:{port}/callback"
    remote_redirect_uri = f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth/callback"

    base_url = f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth"
    params = (
        f"response_type=code"
        f"&client_id=cf-cli-app"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=sha256"
        f"&state={state}"
    )
    local_auth_url = f"{base_url}?{params}&redirect_uri={urllib.parse.quote(local_redirect_uri)}"
    remote_auth_url = f"{base_url}?{params}&redirect_uri={urllib.parse.quote(remote_redirect_uri)}"

    # Start local server
    try:
        httpd = oauth.start_local_server(port)
    except Exception:
        click.echo("❌ Failed to start local server.")
        return None

    if should_attempt_browser_launch():
        # Try to open browser
        click.echo("🌐 Opening browser to sign in to CodeFlash…")
        with contextlib.suppress(Exception):
            webbrowser.open(local_auth_url)

    # Show remote URL and start input thread
    click.echo("\n📋 If browser didn't open, visit this URL:")
    click.echo(f"\n{remote_auth_url}\n")
    click.echo("Paste code here if prompted > ", nl=False)

    # Start thread to wait for manual input
    input_thread = threading.Thread(target=_wait_for_manual_code_input, args=(oauth,))
    input_thread.daemon = True
    input_thread.start()

    waited = 0
    while not oauth.is_complete and waited < 180:
        time.sleep(0.5)
        waited += 0.5

    if not oauth.is_complete:
        httpd.shutdown()
        click.echo("\n❌ Authentication timed out.")
        return None

    # Check which method completed
    api_key = None

    if oauth.manual_code:
        # Manual code was entered
        api_key = oauth.exchange_code_for_token(oauth.manual_code, code_verifier, remote_redirect_uri)
    elif oauth.code:
        # Browser callback received
        if oauth.error or not oauth.state or oauth.state != state:
            httpd.shutdown()
            click.echo("\n❌ Unauthorized.")
            return None

        api_key = oauth.exchange_code_for_token(oauth.code, code_verifier, local_redirect_uri)

    # Cleanup
    time.sleep(3)
    httpd.shutdown()

    if not api_key:
        click.echo("\n❌ Authentication failed.")
    click.echo("\n")
    return api_key
