import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import click
import requests

from codeflash.cli_cmds.console import logger, progress_bar
from codeflash.telemetry.posthog_cf import ph


@lru_cache(maxsize=1)
def get_extension_info() -> dict[str, Any]:
    url = "https://open-vsx.org/api/codeflash/codeflash/latest"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Failed to get extension metadata from open-vsx.org: %s", e)
        return {}


def download_extension_artifacts(vscode_path: Path) -> bool:
    if not (vscode_path / "extensions").exists():
        logger.warning("VSCode extensions directory does not exist")
        return False

    info = get_extension_info()
    download_url = info.get("files", {}).get("download", "")
    latest_version = info.get("version", "")
    if not download_url:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "cf-archive.zip"

        resp = requests.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        extension_path = Path(tmpdir) / "extension"
        if not extension_path.is_dir():
            logger.warning("No 'extension' folder found in archive")
            return False

        dest_path = vscode_path / "extensions" / f"codeflash.codeflash-{latest_version}"

        if dest_path.exists():
            shutil.rmtree(dest_path)  # cleanup if exists
        shutil.copytree(extension_path, dest_path)
    return True


def write_cf_extension_metadata(vscode_path: Path, editor: str, version: str) -> bool:
    metadata_file = vscode_path / "extensions" / "extensions.json"
    if not metadata_file.exists():
        logger.warning("%s extensions metadata file does not exist", editor)
        return False

    data = {
        "identifier": {"id": "codeflash.codeflash", "uuid": "7798581f-9eab-42be-a1b2-87f90973434d"},
        "version": version,
        "location": {"$mid": 1, "path": f"{vscode_path}/extensions/codeflash.codeflash-{version}", "scheme": "file"},
        "relativeLocation": f"codeflash.codeflash-{version}",
        "metadata": {
            "installedTimestamp": int(time.time() * 1000),
            "pinned": False,
            "source": "gallery",
            "id": "7798581f-9eab-42be-a1b2-87f90973434d",
            "publisherId": "bc13551d-2729-4c35-84ce-1d3bd3baab45",
            "publisherDisplayName": "CodeFlash",
            "targetPlatform": "universal",
            "updated": True,
            "isPreReleaseVersion": False,
            "hasPreReleaseVersion": False,
            "isApplicationScoped": False,
            "isMachineScoped": False,
            "isBuiltin": False,
            "private": False,
            "preRelease": False,
        },
    }
    with metadata_file.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata = [entry for entry in metadata if entry.get("identifier", {}).get("id") != data["identifier"]["id"]]
    metadata.append(data)
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return True


def manual_install_vscode_extension(dest_path: Path, editor: str = "VSCode") -> None:
    installed = False
    with progress_bar(f"Manually installing Codeflash for {editor} from open-vsx.org…"):
        try:
            did_download = download_extension_artifacts(dest_path)
            if did_download:
                info = get_extension_info()
                latest_version = info.get("version", "")
                did_write_metadata = write_cf_extension_metadata(dest_path, editor, latest_version)
                if did_write_metadata:
                    installed = True
        except Exception as e:
            logger.error("Failed to install Codeflash for %s: %s", editor, e)
    if installed:
        click.echo(f"✅ Installed the latest version of Codeflash for {editor}.")


def install_vscode_extension() -> None:
    # cursor_path = Path(Path.home()) / ".cursor"
    # windsurf_path = Path(Path.home()) / ".windsurf"

    vscode_path = shutil.which("code")
    if not vscode_path:
        vscode_path = Path(Path.home()) / ".vscode"
        manual_install_vscode_extension(vscode_path)
        return

    error = ""
    with progress_bar("Installing Codeflash for VSCode…"):
        try:
            result = subprocess.run(
                [vscode_path, "--install-extension", "codeflash.codeflash", "--force"],
                check=True,
                text=True,
                timeout=60,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            error = "Installation timed out."
        except subprocess.CalledProcessError as e:
            error = e.stderr or "Unknown error."

    if error:
        ph("vscode-extension-install-failed", {"error": error.strip()})
        click.echo(
            "Failed to install Codeflash for VSCode. Please try installing it manually from the Marketplace: https://marketplace.visualstudio.com/items?itemName=codeflash.codeflash"
        )
        click.echo(error.strip())
    else:
        output = (result.stdout or "").lower()
        if "already installed" in output:
            click.echo("✅ Codeflash for VSCode is already installed.")
            return
        ph("vscode-extension-installed")
        click.echo("✅ Installed the latest version of Codeflash for VSCode.")
