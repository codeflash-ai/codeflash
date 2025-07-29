import json
import tempfile
from pathlib import Path

import pytest
from codeflash.code_utils.checkpoint import CodeflashRunCheckpoint, get_all_historical_functions


class TestCodeflashRunCheckpoint:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_initialization(self, temp_dir):
        module_root = Path("/fake/module/root")
        checkpoint = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)

        # Check if checkpoint file was created
        assert checkpoint.checkpoint_path.exists()

        # Check if metadata was written correctly
        with open(checkpoint.checkpoint_path) as f:
            metadata = json.loads(f.readline())
            assert metadata["type"] == "metadata"
            assert metadata["module_root"] == str(module_root)
            assert "created_at" in metadata
            assert "last_updated" in metadata

    def test_add_function_to_checkpoint(self, temp_dir):
        module_root = Path("/fake/module/root")
        checkpoint = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)

        # Add a function to the checkpoint
        function_name = "module.submodule.function"
        checkpoint.add_function_to_checkpoint(function_name, status="optimized")

        # Read the checkpoint file and verify
        with open(checkpoint.checkpoint_path) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Metadata + function entry

            function_data = json.loads(lines[1])
            assert function_data["type"] == "function"
            assert function_data["function_name"] == function_name
            assert function_data["status"] == "optimized"
            assert "timestamp" in function_data

    def test_add_function_with_additional_info(self, temp_dir):
        module_root = Path("/fake/module/root")
        checkpoint = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)

        # Add a function with additional info
        function_name = "module.submodule.function"
        additional_info = {"execution_time": 1.5, "memory_usage": "10MB"}
        checkpoint.add_function_to_checkpoint(function_name, status="optimized", additional_info=additional_info)

        # Read the checkpoint file and verify
        with open(checkpoint.checkpoint_path) as f:
            lines = f.readlines()
            function_data = json.loads(lines[1])
            assert function_data["execution_time"] == 1.5
            assert function_data["memory_usage"] == "10MB"

    def test_update_metadata_timestamp(self, temp_dir):
        module_root = Path("/fake/module/root")
        checkpoint = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)

        # Get initial timestamp
        with open(checkpoint.checkpoint_path) as f:
            initial_metadata = json.loads(f.readline())
            initial_timestamp = initial_metadata["last_updated"]

        # Wait a bit to ensure timestamp changes
        import time

        time.sleep(0.01)

        # Update timestamp
        checkpoint._update_metadata_timestamp()

        # Check if timestamp was updated
        with open(checkpoint.checkpoint_path) as f:
            updated_metadata = json.loads(f.readline())
            updated_timestamp = updated_metadata["last_updated"]

        assert updated_timestamp > initial_timestamp

    def test_cleanup(self, temp_dir):
        module_root = Path("/fake/module/root")

        # Create multiple checkpoint files
        checkpoint1 = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)
        checkpoint2 = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir)

        # Create a checkpoint for a different module
        different_module = Path("/different/module")
        checkpoint3 = CodeflashRunCheckpoint(different_module, checkpoint_dir=temp_dir)

        # Verify all files exist
        assert checkpoint1.checkpoint_path.exists()
        assert checkpoint2.checkpoint_path.exists()
        assert checkpoint3.checkpoint_path.exists()

        # Clean up files for module_root
        checkpoint1.cleanup()

        # Check that only the files for module_root were deleted
        assert not checkpoint1.checkpoint_path.exists()
        assert not checkpoint2.checkpoint_path.exists()
        assert checkpoint3.checkpoint_path.exists()


class TestGetAllHistoricalFunctions:
    @pytest.fixture
    def setup_checkpoint_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            module_root = Path("/fake/module/root")

            # Create a checkpoint file with some functions
            checkpoint = CodeflashRunCheckpoint(module_root, checkpoint_dir=temp_dir_path)
            checkpoint.add_function_to_checkpoint("module.func1", status="optimized")
            checkpoint.add_function_to_checkpoint("module.func2", status="failed")

            # Create an old checkpoint file (more than 7 days old)
            old_checkpoint_path = temp_dir_path / "codeflash_checkpoint_old.jsonl"
            with open(old_checkpoint_path, "w") as f:
                # Create metadata with old timestamp (8 days ago)
                import time

                old_time = time.time() - (8 * 24 * 60 * 60)
                metadata = {
                    "type": "metadata",
                    "module_root": str(module_root),
                    "created_at": old_time,
                    "last_updated": old_time,
                }
                f.write(json.dumps(metadata) + "\n")

                # Add a function entry
                function_data = {
                    "type": "function",
                    "function_name": "module.old_func",
                    "status": "optimized",
                    "timestamp": old_time,
                }
                f.write(json.dumps(function_data) + "\n")

            # Create a checkpoint for a different module
            different_module = Path("/different/module")
            diff_checkpoint = CodeflashRunCheckpoint(different_module, checkpoint_dir=temp_dir_path)
            diff_checkpoint.add_function_to_checkpoint("different.func", status="optimized")

            yield module_root, temp_dir_path

    def test_get_all_historical_functions(self, setup_checkpoint_files):
        module_root, checkpoint_dir = setup_checkpoint_files

        # Get historical functions
        functions = get_all_historical_functions(module_root, checkpoint_dir)

        # Verify the functions from the current checkpoint are included
        assert "module.func1" in functions
        assert "module.func2" in functions
        assert functions["module.func1"]["status"] == "optimized"
        assert functions["module.func2"]["status"] == "failed"

        # Verify the old function is not included (file should be deleted)
        assert "module.old_func" not in functions

        # Verify the function from the different module is not included
        assert "different.func" not in functions

        # Verify the old checkpoint file was deleted
        assert not (checkpoint_dir / "codeflash_checkpoint_old.jsonl").exists()
