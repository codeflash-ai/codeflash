"""Tests for cleanup of instrumented test files."""

from pathlib import Path
from codeflash.optimization.optimizer import Optimizer


def test_find_leftover_instrumented_test_files_java(tmp_path):
    """Test that Java instrumented test files are detected and can be cleaned up."""
    # Create test directory structure
    test_root = tmp_path / "src" / "test" / "java" / "com" / "example"
    test_root.mkdir(parents=True)

    # Create Java instrumented test files (should be found)
    java_perf1 = test_root / "FibonacciTest__perfinstrumented.java"
    java_perf2 = test_root / "KnapsackTest__perfonlyinstrumented.java"
    java_perf1.touch()
    java_perf2.touch()

    # Create normal Java test file (should NOT be found)
    normal_test = test_root / "CalculatorTest.java"
    normal_test.touch()

    # Find leftover files
    leftover_files = Optimizer.find_leftover_instrumented_test_files(tmp_path)
    leftover_names = {f.name for f in leftover_files}

    # Assert instrumented files are found
    assert "FibonacciTest__perfinstrumented.java" in leftover_names
    assert "KnapsackTest__perfonlyinstrumented.java" in leftover_names

    # Assert normal test file is NOT found
    assert "CalculatorTest.java" not in leftover_names

    # Should find exactly 2 files
    assert len(leftover_files) == 2


def test_find_leftover_instrumented_test_files_python(tmp_path):
    """Test that Python instrumented test files are detected."""
    test_root = tmp_path / "tests"
    test_root.mkdir()

    # Create Python instrumented test files
    py_perf1 = test_root / "test_example__perfinstrumented.py"
    py_perf2 = test_root / "test_foo__perfonlyinstrumented.py"
    py_perf1.touch()
    py_perf2.touch()

    # Create normal Python test file (should NOT be found)
    normal_test = test_root / "test_normal.py"
    normal_test.touch()

    leftover_files = Optimizer.find_leftover_instrumented_test_files(tmp_path)
    leftover_names = {f.name for f in leftover_files}

    assert "test_example__perfinstrumented.py" in leftover_names
    assert "test_foo__perfonlyinstrumented.py" in leftover_names
    assert "test_normal.py" not in leftover_names
    assert len(leftover_files) == 2


def test_find_leftover_instrumented_test_files_javascript(tmp_path):
    """Test that JavaScript/TypeScript instrumented test files are detected."""
    test_root = tmp_path / "tests"
    test_root.mkdir()

    # Create JS/TS instrumented test files
    js_perf1 = test_root / "example__perfinstrumented.test.js"
    ts_perf2 = test_root / "foo__perfonlyinstrumented.spec.ts"
    js_perf1.touch()
    ts_perf2.touch()

    # Create normal test files (should NOT be found)
    normal_test = test_root / "normal.test.js"
    normal_test.touch()

    leftover_files = Optimizer.find_leftover_instrumented_test_files(tmp_path)
    leftover_names = {f.name for f in leftover_files}

    assert "example__perfinstrumented.test.js" in leftover_names
    assert "foo__perfonlyinstrumented.spec.ts" in leftover_names
    assert "normal.test.js" not in leftover_names
    assert len(leftover_files) == 2


def test_find_leftover_instrumented_test_files_mixed(tmp_path):
    """Test that mixed language instrumented test files are all detected."""
    # Create Python dir
    py_dir = tmp_path / "tests"
    py_dir.mkdir()
    (py_dir / "test_foo__perfinstrumented.py").touch()

    # Create Java dir
    java_dir = tmp_path / "src" / "test" / "java"
    java_dir.mkdir(parents=True)
    (java_dir / "FooTest__perfonlyinstrumented.java").touch()

    # Create JS dir
    js_dir = tmp_path / "test"
    js_dir.mkdir()
    (js_dir / "bar__perfinstrumented.test.js").touch()

    # Find all leftover files
    leftover_files = Optimizer.find_leftover_instrumented_test_files(tmp_path)
    leftover_names = {f.name for f in leftover_files}

    # Should find all 3 instrumented files from different languages
    assert "test_foo__perfinstrumented.py" in leftover_names
    assert "FooTest__perfonlyinstrumented.java" in leftover_names
    assert "bar__perfinstrumented.test.js" in leftover_names
    assert len(leftover_files) == 3
