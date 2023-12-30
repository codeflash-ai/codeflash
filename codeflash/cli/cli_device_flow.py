import http.client
import json
import os
import sys
import time
import urllib.parse
from typing import Dict, Any

# Constants
CLIENT_ID = "Iv1.e30ce7aaafaf6412"  # Replace with your GitHub App's client ID


# Function to parse API responses
def parse_response(response: http.client.HTTPResponse) -> Dict[str, Any]:
    if response.status in [200, 201]:
        return json.loads(response.read().decode())
    else:
        print(response.read().decode())
        sys.exit(1)


# Function to request a device code
def request_device_code() -> Dict[str, Any]:
    conn = http.client.HTTPSConnection("github.com")
    payload = urllib.parse.urlencode({"client_id": CLIENT_ID})
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    conn.request("POST", "/login/device/code", payload, headers)
    return parse_response(conn.getresponse())


# Function to request a token
def request_token(device_code: str) -> Dict[str, Any]:
    conn = http.client.HTTPSConnection("github.com")
    payload = urllib.parse.urlencode({
        "client_id": CLIENT_ID,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code"
    })
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'application/json'}
    conn.request("POST", "/login/oauth/access_token", payload, headers)
    return parse_response(conn.getresponse())


# Function to poll for a token
def poll_for_token(device_code: str, interval: int):
    while True:
        response = request_token(device_code)
        error, access_token = response.get("error"), response.get("access_token")

        if error:
            if error == "authorization_pending":
                time.sleep(interval)
                continue
            elif error == "slow_down":
                time.sleep(interval + 5)
                continue
            elif error == "expired_token":
                print("The device code has expired. Please run `login` again.")
                sys.exit(1)
            elif error == "access_denied":
                print("Login cancelled by user.")
                sys.exit(1)
            else:
                print(response)
                sys.exit(1)

        with open(".token", "w") as token_file:
            token_file.write(access_token)
        os.chmod(".token", 0o600)
        break


# Login function
def login():
    response = request_device_code()
    verification_uri, user_code, device_code, interval = response.values_at("verification_uri", "user_code",
                                                                            "device_code", "interval")
    print(f"Please visit: {verification_uri} and enter code: {user_code}")
    poll_for_token(device_code, interval)
    print("Successfully authenticated!")


# Main function
def main():
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        print("usage: codeflash <login | help>")
    elif sys.argv[1] == "login":
        login()
    else:
        print(f"Unknown command {sys.argv[1]}")


if __name__ == "__main__":
    main()
