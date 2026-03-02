from playwright.sync_api import sync_playwright
import subprocess
import sys


def ensure_browser_installed():
    try:
        with sync_playwright() as p:
            p.chromium.launch(headless=True)
    except Exception:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True
        )


ensure_browser_installed()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    page.set_content("""
        <div id="root"></div>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script>
            const root = ReactDOM.createRoot(document.getElementById('root'));
            root.render(React.createElement("h1", null, "Hello from React"));
        </script>
    """)

    print(page.content())