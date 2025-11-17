from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import socket
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
                oauth_handler.code = params.get("code", [None])[0]
                oauth_handler.state = params.get("state", [None])[0]
                oauth_handler.error = params.get("error", [None])[0]
                oauth_handler.theme = params.get("theme", ["light"])[0]

                # Send HTML response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                html_content = self._get_html_response()
                self.wfile.write(html_content.encode())

                oauth_handler.is_complete = True

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

    def wait_for_callback(self, httpd: http.server.HTTPServer, timeout: int = 120) -> bool:  # noqa: ARG002
        """Wait for OAuth callback with timeout."""
        waited = 0
        while not self.is_complete and waited < timeout:
            time.sleep(0.5)
            waited += 0.5

        return self.is_complete

    def exchange_code_for_token(self, code: str, code_verifier: str, redirect_uri: str) -> str | None:
        """Exchange authorization code for API token."""
        token_url = f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "client_id": "cf_vscode_app",
        }

        try:
            resp = requests.post(
                token_url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=10
            )
            resp.raise_for_status()
            token_json = resp.json()
            api_key = token_json.get("access_token")

            if not api_key:
                self.token_error = "No access token in response"  # noqa: S105
                return None

        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error_description", error_data.get("error", error_msg))
            except Exception:  # noqa: S110
                pass
            self.token_error = "Unauthorized"  # noqa: S105
            click.echo(f"❌ {self.token_error}")
            return None
        except Exception:
            self.token_error = "Unauthorized"  # noqa: S105
            click.echo(f"❌ {self.token_error}")
            return None
        else:
            return api_key


def perform_oauth_signin() -> str | None:
    """Perform OAuth PKCE flow and return API key if successful.

    Returns None if failed.
    """
    oauth = OAuthHandler()

    # Setup PKCE
    port = oauth.get_free_port()
    redirect_uri = f"http://localhost:{port}/callback"
    code_verifier, code_challenge = oauth.generate_pkce_pair()
    state = "".join(secrets.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(16))

    # Build authorization URL
    auth_url = (
        f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth?"
        f"response_type=code"
        f"&client_id=cf_vscode_app"
        f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method=sha256"
        f"&state={state}"
    )

    # Start local server
    httpd = oauth.start_local_server(port)

    # Open browser
    click.echo("🌐 Opening browser to sign in to CodeFlash…")
    webbrowser.open(auth_url)

    click.echo(f"\n📋 If your browser didn't open, visit: {auth_url}\n")

    # Wait for callback
    click.echo("⏳ Waiting for authentication...")
    success = oauth.wait_for_callback(httpd, timeout=180)

    if not success:
        httpd.shutdown()
        click.echo("❌ Authentication timed out. Please try again.")
        return None

    if oauth.error:
        httpd.shutdown()
        click.echo("❌ Authentication failed:")
        return None

    if not oauth.code or not oauth.state:
        httpd.shutdown()
        click.echo("❌ Unauthorized.")
        return None

    if oauth.state != state:
        httpd.shutdown()
        click.echo("❌ Unauthorized.")
        return None

    api_key = oauth.exchange_code_for_token(oauth.code, code_verifier, redirect_uri)

    # Wait for browser to poll status
    time.sleep(3)

    # Shutdown server
    httpd.shutdown()

    return api_key
