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

