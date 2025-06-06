from __future__ import annotations

import datetime
import json
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import click

if TYPE_CHECKING:
    import argparse


class CodeflashRunCheckpoint:
    def __init__(self, module_root: Path, checkpoint_dir: Path = Path("/tmp")) -> None:  # noqa: S108
        self.module_root = module_root
        self.checkpoint_dir = Path(checkpoint_dir)
        # Create a unique checkpoint file name
        unique_id = str(uuid.uuid4())[:8]
        checkpoint_filename = f"codeflash_checkpoint_{unique_id}.jsonl"
        self.checkpoint_path = self.checkpoint_dir / checkpoint_filename

        # Initialize the checkpoint file with metadata
        self._initialize_checkpoint_file()

    def _initialize_checkpoint_file(self) -> None:
        """Create a new checkpoint file with metadata."""
        metadata = {
            "type": "metadata",
            "module_root": str(self.module_root),
            "created_at": time.time(),
            "last_updated": time.time(),
        }

        with self.checkpoint_path.open("w") as f:
            f.write(json.dumps(metadata) + "\n")

    def add_function_to_checkpoint(
        self,
        function_fully_qualified_name: str,
        status: str = "optimized",
        additional_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a function to the checkpoint after it has been processed.

        Args:
        ----
            function_fully_qualified_name: The fully qualified name of the function
            status: Status of optimization (e.g., "optimized", "failed", "skipped")
            additional_info: Any additional information to store about the function

        """
        if additional_info is None:
            additional_info = {}

        function_data = {
            "type": "function",
            "function_name": function_fully_qualified_name,
            "status": status,
            "timestamp": time.time(),
            **additional_info,
        }

        with self.checkpoint_path.open("a") as f:
            f.write(json.dumps(function_data) + "\n")

        # Update the metadata last_updated timestamp
        self._update_metadata_timestamp()

    def _update_metadata_timestamp(self) -> None:
        """Update the last_updated timestamp in the metadata."""
        # Read the first line (metadata)
        with self.checkpoint_path.open() as f:
            metadata = json.loads(f.readline())
            rest_content = f.read()

        # Update the timestamp
        metadata["last_updated"] = time.time()

        # Write all lines to a temporary file

        with self.checkpoint_path.open("w") as f:
            f.write(json.dumps(metadata) + "\n")
            f.write(rest_content)

    def cleanup(self) -> None:
        """Unlink all the checkpoint files for this module_root."""
        to_delete = []
        self.checkpoint_path.unlink(missing_ok=True)

        for file in self.checkpoint_dir.glob("codeflash_checkpoint_*.jsonl"):
            with file.open() as f:
                # Skip the first line (metadata)
                first_line = next(f)
                metadata = json.loads(first_line)
                if metadata.get("module_root", str(self.module_root)) == str(self.module_root):
                    to_delete.append(file)
        for file in to_delete:
            file.unlink(missing_ok=True)


def get_all_historical_functions(module_root: Path, checkpoint_dir: Path) -> dict[str, dict[str, str]]:
    """Get information about all processed functions, regardless of status.

    Returns
    -------
        Dictionary mapping function names to their processing information

    """
    processed_functions = {}
    to_delete = []

    for file in checkpoint_dir.glob("codeflash_checkpoint_*.jsonl"):
        with file.open() as f:
            # Skip the first line (metadata)
            first_line = next(f)
            metadata = json.loads(first_line)
            if metadata.get("last_updated"):
                last_updated = datetime.datetime.fromtimestamp(metadata["last_updated"])  # noqa: DTZ006
                if datetime.datetime.now() - last_updated >= datetime.timedelta(days=7):  # noqa: DTZ005
                    to_delete.append(file)
                    continue
            if metadata.get("module_root") != str(module_root):
                continue

            for line in f:
                entry = json.loads(line)
                if entry.get("type") == "function":
                    processed_functions[entry["function_name"]] = entry
    for file in to_delete:
        file.unlink(missing_ok=True)
    return processed_functions


def ask_should_use_checkpoint_get_functions(args: argparse.Namespace) -> Optional[dict[str, dict[str, str]]]:
    previous_checkpoint_functions = None
    if args.all and (sys.platform == "linux" or sys.platform == "darwin") and Path("/tmp").is_dir():  # noqa: S108 #TODO: use the temp dir from codeutils-compat.py
        previous_checkpoint_functions = get_all_historical_functions(args.module_root, Path("/tmp"))  # noqa: S108
        if previous_checkpoint_functions and click.confirm(
            "Previous Checkpoint detected from an incomplete optimization run, shall I continue the optimization from that point?",
            default=True,
        ):
            pass
        else:
            previous_checkpoint_functions = None
    return previous_checkpoint_functions
