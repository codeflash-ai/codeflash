import json
import shutil
import tempfile
import time
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from codeflash.cli_cmds.console import logger, progress_bar

supported_editor_paths = [
    (Path(Path.home()) / ".vscode", "VSCode"),
    (Path(Path.home()) / ".cursor", "Cursor"),
    (Path(Path.home()) / ".windsurf", "Windsurf"),
]


@lru_cache(maxsize=1)
def get_extension_info() -> dict[str, Any]:
    url = "https://open-vsx.org/api/codeflash/codeflash/latest"
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error("Failed to retrieve extension metadata from open-vsx.org: %s", e)
        return {}


@contextmanager
def download_and_extract_extension(download_url: str) -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "extension.zip"

        resp = requests.get(download_url, stream=True, timeout=60)
        resp.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)

        extension_path = tmpdir_path / "extension"
        if not extension_path.is_dir():
            raise FileNotFoundError("Extension folder not found in downloaded archive")

        yield extension_path


@contextmanager
def download_and_extract_extension_with_progress(download_url: str) -> Path:
    with (
        progress_bar("Downloading CodeFlash extension from open-vsx.org..."),
        download_and_extract_extension(download_url) as extension_path,
    ):
        yield extension_path


def copy_extension_artifacts(src: Path, dest: Path, version: str) -> bool:
    dst_extensions_dir = dest / "extensions"
    if not dst_extensions_dir.exists():
        logger.warning("Extensions directory does not exist: %s", str(dst_extensions_dir))
        return False

    dest_path = dst_extensions_dir / f"codeflash.codeflash-{version}"

    shutil.copytree(src, dest_path, dirs_exist_ok=True)
    return True


def get_metadata_file_path(editor_path: Path) -> Path:
    return editor_path / "extensions" / "extensions.json"


@lru_cache(maxsize=len(supported_editor_paths))
def get_cf_extension_metadata(editor_path: Path) -> list[dict[str, Any]]:
    metadata_file = get_metadata_file_path(editor_path)
    if not metadata_file.exists():
        logger.warning("Extensions metadata file does not exist")
        return []
    with metadata_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_cf_extension_metadata(editor_path: Path, version: str) -> bool:
    data = {
        "identifier": {"id": "codeflash.codeflash", "uuid": "7798581f-9eab-42be-a1b2-87f90973434d"},
        "version": version,
        "location": {"$mid": 1, "path": f"{editor_path}/extensions/codeflash.codeflash-{version}", "scheme": "file"},
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
    installed_extensions = get_cf_extension_metadata(editor_path)
    if not installed_extensions:
        return False
    installed_extensions = [
        ext for ext in installed_extensions if ext.get("identifier", {}).get("id") != data["identifier"]["id"]
    ]
    installed_extensions.append(data)
    with get_metadata_file_path(editor_path).open("w", encoding="utf-8") as f:
        json.dump(installed_extensions, f)
    return True


def is_latest_version_installed(editor_path: Path, latest_version: str) -> bool:
    installed_extensions = get_cf_extension_metadata(editor_path)
    current_version = ""
    for ext in installed_extensions:
        if ext.get("identifier", {}).get("id") == "codeflash.codeflash":
            current_version = ext.get("version", "")
            break
    return current_version == latest_version


def manually_install_vscode_extension(downloadable_paths: list[tuple[Path, str]]) -> None:
    with progress_bar("Fetching extension metadata..."):
        info = get_extension_info()

    download_url = info.get("files", {}).get("download", "")
    latest_version = info.get("version", "")

    if not download_url or not latest_version:
        logger.error("Failed to retrieve extension metadata")
        return

    successful_installs = []
    with download_and_extract_extension_with_progress(download_url) as extension_path:
        for editor_path, editor in downloadable_paths:
            try:
                did_copy = copy_extension_artifacts(extension_path, editor_path, latest_version)
                if not did_copy:
                    continue
                did_write_metadata = write_cf_extension_metadata(editor_path, latest_version)
                if not did_write_metadata:
                    continue

                successful_installs.append(editor)
            except Exception as e:
                logger.error("Failed to install CodeFlash extension for %s: %s", editor, e)
    if successful_installs:
        logger.info("Successfully installed CodeFlash extension for: %s", ", ".join(successful_installs))


def install_vscode_extension() -> None:
    editors_installed = []
    downloadable_paths = []

    for editor_path, editor in supported_editor_paths:
        if not editor_path.exists():
            continue

        editors_installed.append(editor)

        info = get_extension_info()
        latest_version = info.get("version", "")

        if not latest_version or is_latest_version_installed(editor_path, latest_version):
            continue

        downloadable_paths.append((editor_path, editor))

    if not downloadable_paths:
        if editors_installed:
            logger.info("CodeFlash extension is already installed and up-to-date for: %s", ", ".join(editors_installed))
            return

        logger.info("No supported editors found for CodeFlash extension installation")
        return

    downloadable_editors = ", ".join([editor for _, editor in downloadable_paths])
    logger.info("Installing CodeFlash extension for %s...", downloadable_editors)
    manually_install_vscode_extension(downloadable_paths)
