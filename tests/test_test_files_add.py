from pathlib import Path

from codeflash.models.models import TestFile, TestFiles
from codeflash.models.test_type import TestType


class TestTestFilesAdd:
    def test_add_unique_test_file(self) -> None:
        tf = TestFiles(test_files=[])
        test_file = TestFile(
            instrumented_behavior_file_path=Path("/tmp/test_behavior.py"),
            benchmarking_file_path=Path("/tmp/test_perf.py"),
            test_type=TestType.GENERATED_REGRESSION,
        )
        tf.add(test_file)
        assert len(tf.test_files) == 1
        assert tf.test_files[0] is test_file

    def test_add_duplicate_is_noop(self) -> None:
        tf = TestFiles(test_files=[])
        test_file = TestFile(
            instrumented_behavior_file_path=Path("/tmp/test_behavior.py"),
            benchmarking_file_path=Path("/tmp/test_perf.py"),
            test_type=TestType.GENERATED_REGRESSION,
        )
        tf.add(test_file)
        tf.add(test_file)  # silent skip — first write wins
        assert len(tf.test_files) == 1

    def test_add_many_files_performance(self) -> None:
        tf = TestFiles(test_files=[])
        for i in range(100):
            test_file = TestFile(
                instrumented_behavior_file_path=Path(f"/tmp/test_behavior_{i}.py"),
                benchmarking_file_path=Path(f"/tmp/test_perf_{i}.py"),
                test_type=TestType.GENERATED_REGRESSION,
            )
            tf.add(test_file)

        assert len(tf.test_files) == 100
        assert len(tf._seen_paths) == 100
        # Verify all paths are unique in the set
        expected_paths = {Path(f"/tmp/test_behavior_{i}.py") for i in range(100)}
        assert tf._seen_paths == expected_paths
