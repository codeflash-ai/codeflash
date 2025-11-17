import threading
import socket
import http.server
import urllib.parse
import random
import string
import base64
import hashlib
import time
import json
import webbrowser
import requests
from typing import Optional, Tuple
import click

from codeflash.api.cfapi import get_cfapi_base_urls


class OAuthHandler:
    """Handles OAuth PKCE flow for CodeFlash authentication"""

    def __init__(self):
        self.code: Optional[str] = None
        self.state: Optional[str] = None
        self.error: Optional[str] = None
        self.is_complete = False
        self.token_error: Optional[str] = None

    def create_callback_handler(self):
        """Creates HTTP handler for OAuth callback"""
        oauth_handler = self

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            server_version = "CFHTTP"

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)

                if parsed.path == "/status":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    status = {
                        "success": oauth_handler.token_error is None and oauth_handler.code is not None,
                        "error": oauth_handler.token_error
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

                # Send HTML response
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()

                html_content = self._get_html_response()
                self.wfile.write(html_content.encode())

                oauth_handler.is_complete = True

            def _get_html_response(self):
                """Returns simple HTML response"""
                if oauth_handler.error:
                    return self._get_error_html(oauth_handler.error)
                elif oauth_handler.code:
                    return self._get_loading_html()
                else:
                    return self._get_error_html("unauthorized")

            @staticmethod
            def _get_loading_html():
                """Loading state while exchanging token"""
                return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeFlash Authentication</title>
    <style>
        body {
            font-family: -apple-system, system-ui, sans-serif;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }
        .card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            padding: 40px;
            max-width: 400px;
            text-align: center;
        }
        .icon {
            font-size: 48px;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 24px;
            margin: 0 0 10px;
            color: #212529;
        }
        p {
            color: #6c757d;
            margin: 0;
        }
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid #e9ecef;
            border-top-color: #0d6efd;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .error-box {
            background: #f8d7da;
            color: #842029;
            padding: 12px;
            border-radius: 6px;
            margin-top: 20px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="card" id="content">
        <div class="spinner"></div>
        <h1>Authenticating</h1>
        <p>Please wait...</p>
    </div>

    <script>
        let pollCount = 0;
        const maxPolls = 60;

        function checkStatus() {
            fetch('/status')
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        showSuccess();
                    } else if (data.error) {
                        showError(data.error);
                    } else if (pollCount < maxPolls) {
                        pollCount++;
                        setTimeout(checkStatus, 500);
                    } else {
                        showError('Authentication timed out');
                    }
                })
                .catch(() => {
                    if (pollCount < maxPolls) {
                        pollCount++;
                        setTimeout(checkStatus, 500);
                    }
                });
        }

        function showSuccess() {
            document.getElementById('content').innerHTML = `
                <h1>Success!</h1>
                <p>You can close this window.</p>
            `;
        }

        function showError(message) {
            document.getElementById('content').innerHTML = `
                <h1>Authentication Failed</h1>
                <div class="error-box">${message}</div>
            `;
        }

        setTimeout(checkStatus, 1000);
    </script>
</body>
</html>
                """

            @staticmethod
            def _get_error_html(self, error_message: str):
                """Error state HTML"""
                return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeFlash Authentication</title>
    <style>
        body {{
            font-family: -apple-system, system-ui, sans-serif;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
        }}
        .card {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            padding: 40px;
            max-width: 400px;
            text-align: center;
        }}
        h1 {{
            font-size: 24px;
            margin: 0 0 10px;
            color: #212529;
        }}
        .error-box {{
            background: #f8d7da;
            color: #842029;
            padding: 12px;
            border-radius: 6px;
            margin-top: 20px;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Authentication Failed</h1>
        <div class="error-box">{error_message}</div>
    </div>
</body>
</html>
                """

            def log_message(self, format, *args):
                pass

        return CallbackHandler

    @staticmethod
    def get_free_port() -> int:
        """Find an available port"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @staticmethod
    def generate_pkce_pair() -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        code_verifier = ''.join(
            random.choices(string.ascii_letters + string.digits + "-._~", k=64)
        )
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b'=').decode()
        return code_verifier, code_challenge

    def start_local_server(self, port: int) -> http.server.HTTPServer:
        """Start local HTTP server for OAuth callback"""
        handler_class = self.create_callback_handler()
        httpd = http.server.HTTPServer(("localhost", port), handler_class)

        def serve_forever_wrapper():
            httpd.serve_forever()

        server_thread = threading.Thread(target=serve_forever_wrapper)
        server_thread.daemon = True
        server_thread.start()

        return httpd

    def wait_for_callback(self, httpd: http.server.HTTPServer, timeout: int = 120) -> bool:
        """Wait for OAuth callback with timeout"""
        waited = 0
        while not self.is_complete and waited < timeout:
            time.sleep(0.5)
            waited += 0.5

        return self.is_complete

    def exchange_code_for_token(
        self,
        code: str,
        code_verifier: str,
        redirect_uri: str
    ) -> Optional[str]:
        """Exchange authorization code for API token"""
        token_url = f"{get_cfapi_base_urls().cfwebapp_base_url}/codeflash/auth/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": code_verifier,
            "redirect_uri": redirect_uri,
            "client_id": "cf_vscode_app"
        }

        try:
            resp = requests.post(
                token_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(data),
                timeout=10,
            )
            resp.raise_for_status()
            token_json = resp.json()
            api_key = token_json.get("access_token")

            if not api_key:
                self.token_error = "No access token in response"
                return None

            return api_key
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error_description", error_data.get("error", error_msg))
            except:
                pass
            self.token_error = "Unauthorized"
            click.echo(f"❌ {self.token_error}")
            return None
        except Exception as e:
            self.token_error = "Unauthorized"
            click.echo(f"❌ {self.token_error}")
            return None


def perform_oauth_signin() -> Optional[str]:
    """
    Perform OAuth PKCE flow and return API key if successful.
    Returns None if failed.
    """
    oauth = OAuthHandler()

    # Setup PKCE
    port = oauth.get_free_port()
    redirect_uri = f"http://localhost:{port}/callback"
    code_verifier, code_challenge = oauth.generate_pkce_pair()
    state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

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
    success = oauth.wait_for_callback(httpd, timeout=120)

    if not success:
        httpd.shutdown()
        click.echo("❌ Authentication timed out. Please try again.")
        return None

    if oauth.error:
        httpd.shutdown()
        click.echo(f"❌ Authentication failed:")
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
