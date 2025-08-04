"""Tests for unused helper function revert functionality."""

import tempfile
from pathlib import Path

import pytest
from codeflash.context.unused_definition_remover import detect_unused_helper_functions
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeStringsMarkdown, get_code_block_splitter
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


@pytest.fixture
def temp_project():
    """Create a temporary project with test files."""
    temp_dir = Path(tempfile.mkdtemp())

    # Main file with function that calls helpers
    main_file = temp_dir / "main.py"
    main_file.write_text("""
def entrypoint_function(n):
    \"\"\"Function that calls two helper functions.\"\"\"
    result1 = helper_function_1(n)
    result2 = helper_function_2(n)
    return result1 + result2

def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function.\"\"\"
    return x * 3
""")

    # Create test config
    test_cfg = TestConfig(
        tests_root=temp_dir / "tests",
        tests_project_rootdir=temp_dir,
        project_root_path=temp_dir,
        test_framework="pytest",
        pytest_cmd="pytest",
    )

    yield temp_dir, main_file, test_cfg

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_detect_unused_helper_functions(temp_project):
    """Test that unused helper functions are correctly detected."""
    temp_dir, main_file, test_cfg = temp_project

    # Optimized version that only calls one helper
    optimized_code = f"""
{get_code_block_splitter("main.py")}
def entrypoint_function(n):
    \"\"\"Optimized function that only calls one helper.\"\"\"
    result1 = helper_function_1(n)
    return result1 + n * 3  # Inlined helper_function_2

def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function - MODIFIED VERSION should be reverted.\"\"\"
    return x * 4  # This change should be reverted to original x * 3
"""

    # Create FunctionToOptimize instance
    function_to_optimize = FunctionToOptimize(
        file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
    )

    # Create function optimizer
    optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=test_cfg,
        function_to_optimize_source_code=main_file.read_text(),
    )

    # Get original code context to find helper functions
    ctx_result = optimizer.get_code_optimization_context()
    assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

    code_context = ctx_result.unwrap()

    # Test unused helper detection
    unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

    # Should detect helper_function_2 as unused
    unused_names = {uh.qualified_name for uh in unused_helpers}
    expected_unused = {"helper_function_2"}

    assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

    # Also test the complete replace_function_and_helpers_with_optimized_code workflow
    # First modify the optimized code to include a MODIFIED unused helper
    optimized_code_with_modified_helper = f"""
{get_code_block_splitter("main.py")}
def entrypoint_function(n):
    \"\"\"Optimized function that only calls one helper.\"\"\"
    result1 = helper_function_1(n)
    return result1 + n * 3  # Inlined helper_function_2

def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function - MODIFIED VERSION should be reverted.\"\"\"
    return x * 7  # This should be reverted to x * 3
"""

    original_helper_code = {main_file: main_file.read_text()}

    # Apply optimization and test reversion
    optimizer.replace_function_and_helpers_with_optimized_code(
        code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code_with_modified_helper), original_helper_code
    )
    # Check final file content
    final_content = main_file.read_text()

    # The entrypoint should be optimized
    assert "result1 + n * 3" in final_content, "Entrypoint function should be optimized"

    # helper_function_2 should be reverted to original (x * 3, NOT the modified x * 7)
    assert "return x * 3" in final_content, "helper_function_2 should be reverted to original"
    assert "return x * 7" not in final_content, "helper_function_2 should NOT contain the modified version"

    # helper_function_1 should remain (it's still called)
    assert "def helper_function_1(x):" in final_content, "helper_function_1 should still exist"

    # Also test the complete replace_function_and_helpers_with_optimized_code workflow
    original_helper_code = {main_file: main_file.read_text()}

    # Apply optimization and test reversion
    optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

    # Check final file content
    final_content = main_file.read_text()

    # The entrypoint should be optimized
    assert "result1 + n * 3" in final_content, "Entrypoint function should be optimized"

    # helper_function_2 should be reverted to original (return x * 3, NOT the modified x * 4)
    assert "return x * 3" in final_content, "helper_function_2 should be reverted to original"
    assert "return x * 4" not in final_content, "helper_function_2 should NOT contain the modified version"

    # helper_function_1 should remain as optimized (it's still called)
    assert "def helper_function_1(x):" in final_content, "helper_function_1 should still exist"


def test_revert_unused_helper_functions(temp_project):
    """Test that unused helper functions are correctly reverted to original definitions."""
    temp_dir, main_file, test_cfg = temp_project

    # Optimized version that only calls one helper and modifies the unused one
    optimized_code = f"""
{get_code_block_splitter("main.py") }
def entrypoint_function(n):
    \"\"\"Optimized function that only calls one helper.\"\"\"
    result1 = helper_function_1(n)
    return result1 + n * 3  # Inlined helper_function_2

def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Modified helper function - should be reverted.\"\"\"
    return x * 4  # This change should be reverted
"""

    # Create FunctionToOptimize instance
    function_to_optimize = FunctionToOptimize(
        file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
    )

    # Create function optimizer
    optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=test_cfg,
        function_to_optimize_source_code=main_file.read_text(),
    )

    # Get original code context
    ctx_result = optimizer.get_code_optimization_context()
    assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

    code_context = ctx_result.unwrap()

    # Store original helper code
    original_helper_code = {main_file: main_file.read_text()}
    original_content = main_file.read_text()

    # Test the new functionality - this should:
    # 1. Apply the optimization
    # 2. Detect unused helpers
    # 3. Revert unused helpers to original definitions
    optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

    # Check final file content
    final_content = main_file.read_text()

    # The entrypoint should be optimized (inline the helper_function_2 call)
    assert "result1 + n * 3" in final_content, "Entrypoint function should be optimized"

    # helper_function_2 should be reverted to original (return x * 3, not x * 4)
    assert "return x * 3" in final_content, "helper_function_2 should be reverted to original"
    assert "return x * 4" not in final_content, "helper_function_2 should not contain the optimized version"

    # helper_function_1 should remain as optimized (it's still called)
    assert "def helper_function_1(x):" in final_content, "helper_function_1 should still exist"


def test_no_unused_helpers_no_revert(temp_project):
    """Test that when all helpers are still used, nothing is reverted."""
    temp_dir, main_file, test_cfg = temp_project

    # Optimized version that still calls both helpers
    optimized_code = f"""
{get_code_block_splitter("main.py")}
def entrypoint_function(n):
    \"\"\"Optimized function that still calls both helpers.\"\"\"
    result1 = helper_function_1(n)
    result2 = helper_function_2(n)
    return result1 + result2  # Still using both

def helper_function_1(x):
    \"\"\"First helper function - optimized.\"\"\"
    return x << 1  # Optimized to use bit shift

def helper_function_2(x):
    \"\"\"Second helper function - optimized.\"\"\"
    return x * 3
"""

    # Create FunctionToOptimize instance
    function_to_optimize = FunctionToOptimize(
        file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
    )

    # Create function optimizer
    optimizer = FunctionOptimizer(
        function_to_optimize=function_to_optimize,
        test_cfg=test_cfg,
        function_to_optimize_source_code=main_file.read_text(),
    )

    # Get original code context
    ctx_result = optimizer.get_code_optimization_context()
    assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

    code_context = ctx_result.unwrap()

    # Store original helper code
    original_helper_code = {main_file: main_file.read_text()}

    # Test detection - should find no unused helpers
    unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)
    assert len(unused_helpers) == 0, "No helpers should be detected as unused"

    # Apply optimization
    optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

    # Check final file content - should contain the optimized versions
    final_content = main_file.read_text()

    # Both helpers should be optimized
    assert "x << 1" in final_content, "helper_function_1 should be optimized to use bit shift"
    assert "result1 + result2" in final_content, "Entrypoint should still call both helpers"


def test_detect_unused_in_multi_file_project():
    """Test detection of unused helpers across multiple files."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file
        main_file = temp_dir / "main.py"
        main_file.write_text("""
from helpers import helper_function_1, helper_function_2

def entrypoint_function(n):
    \"\"\"Function that calls helpers from another file.\"\"\"
    result1 = helper_function_1(n)
    result2 = helper_function_2(n)
    return result1 + result2
""")

        # Helper file
        helper_file = temp_dir / "helpers.py"
        helper_file.write_text("""
def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function.\"\"\"
    return x * 3
""")

        # Optimized version that only calls one helper
        optimized_code = f"""
{get_code_block_splitter("main.py")}
from helpers import helper_function_1

def entrypoint_function(n):
    \"\"\"Optimized function that only calls one helper.\"\"\"
    result1 = helper_function_1(n)
    return result1 + n * 3  # Inlined helper_function_2
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Create FunctionToOptimize instance
        function_to_optimize = FunctionToOptimize(
            file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection
        unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

        # Should detect helper_function_2 as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}
        expected_unused = {"helper_function_2"}

        assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # First, simulate modified helper in the helper file
        helper_file.write_text("""
def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function - MODIFIED VERSION.\"\"\"
    return x * 9  # This should be reverted to x * 3
""")

        # Store original helper code (before modification)
        original_helper_code = {
            main_file: """
from helpers import helper_function_1, helper_function_2

def entrypoint_function(n):
    \"\"\"Function that calls helpers from another file.\"\"\"
    result1 = helper_function_1(n)
    result2 = helper_function_2(n)
    return result1 + result2
""",
            helper_file: """
def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function.\"\"\"
    return x * 3
""",
        }

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)
        # Check main file content
        main_content = main_file.read_text()
        assert "result1 + n * 3" in main_content, "Entrypoint function should be optimized"
        assert "from helpers import helper_function_1" in main_content, "Import should be updated"

        # Check helper file content - helper_function_2 should be reverted to original
        helper_content = helper_file.read_text()
        assert "def helper_function_1(x):" in helper_content, "helper_function_1 should still exist"
        assert "def helper_function_2(x):" in helper_content, "helper_function_2 should exist"
        assert "return x * 3" in helper_content, "helper_function_2 should be reverted to original"
        assert "return x * 9" not in helper_content, "helper_function_2 should NOT contain the modified version"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # First, simulate modified helper in the helper file
        helper_file.write_text("""
def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function - MODIFIED VERSION.\"\"\"
    return x * 5  # This should be reverted to x * 3
""")

        # Store original helper code (before modification)
        original_helper_code = {
            main_file: """
from helpers import helper_function_1, helper_function_2

def entrypoint_function(n):
    \"\"\"Function that calls helpers from another file.\"\"\"
    result1 = helper_function_1(n)
    result2 = helper_function_2(n)
    return result1 + result2
""",
            helper_file: """
def helper_function_1(x):
    \"\"\"First helper function.\"\"\"
    return x * 2

def helper_function_2(x):
    \"\"\"Second helper function.\"\"\"
    return x * 3
""",
        }

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

        # Check main file content
        main_content = main_file.read_text()
        assert "result1 + n * 3" in main_content, "Entrypoint function should be optimized"
        assert "from helpers import helper_function_1" in main_content, "Import should be updated"

        # Check helper file content - helper_function_2 should be reverted to original
        helper_content = helper_file.read_text()
        assert "def helper_function_1(x):" in helper_content, "helper_function_1 should still exist"
        assert "def helper_function_2(x):" in helper_content, "helper_function_2 should exist"
        assert "return x * 3" in helper_content, "helper_function_2 should be reverted to original"
        assert "return x * 5" not in helper_content, "helper_function_2 should NOT contain the modified version"

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_class_method_entrypoint_with_helper_methods():
    """Test unused helper detection when entrypoint is a class method that calls other methods."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file with class containing methods
        main_file = temp_dir / "main.py"
        main_file.write_text("""
class Calculator:
    def entrypoint_method(self, n):
        \"\"\"Main method that calls helper methods.\"\"\"
        result1 = self.helper_method_1(n)
        result2 = self.helper_method_2(n)
        return result1 + result2

    def helper_method_1(self, x):
        \"\"\"First helper method.\"\"\"
        return x * 2

    def helper_method_2(self, x):
        \"\"\"Second helper method.\"\"\"
        return x * 3
""")

        # Optimized version that only calls one helper method
        optimized_code = f"""
{get_code_block_splitter("main.py") }
class Calculator:
    def entrypoint_method(self, n):
        \"\"\"Optimized method that only calls one helper.\"\"\"
        result1 = self.helper_method_1(n)
        return result1 + n * 3  # Inlined helper_method_2

    def helper_method_1(self, x):
        \"\"\"First helper method.\"\"\"
        return x * 2

    def helper_method_2(self, x):
        \"\"\"Second helper method - should be reverted.\"\"\"
        return x * 4
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Create FunctionToOptimize instance for class method
        from codeflash.models.models import FunctionParent

        function_to_optimize = FunctionToOptimize(
            file_path=main_file,
            function_name="entrypoint_method",
            qualified_name="Calculator.entrypoint_method",
            parents=[FunctionParent(name="Calculator", type="ClassDef")],
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection
        unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

        # Should detect Calculator.helper_method_2 as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}
        expected_unused = {"Calculator.helper_method_2"}

        assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # Update optimized code to include a MODIFIED unused helper
        optimized_code_with_modified_helper = f"""
{get_code_block_splitter("main.py")}
class Calculator:
    def entrypoint_method(self, n):
        \"\"\"Optimized method that only calls one helper.\"\"\"
        result1 = self.helper_method_1(n)
        return result1 + n * 3  # Inlined helper_method_2

    def helper_method_1(self, x):
        \"\"\"First helper method.\"\"\"
        return x * 2

    def helper_method_2(self, x):
        \"\"\"Second helper method - MODIFIED VERSION should be reverted.\"\"\"
        return x * 8  # This should be reverted to x * 3
"""

        original_helper_code = {main_file: main_file.read_text()}

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(
            code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code_with_modified_helper), original_helper_code
        )

        # Check final file content
        final_content = main_file.read_text()

        # The entrypoint method should be optimized
        assert "result1 + n * 3" in final_content, "Entrypoint method should be optimized"

        # helper_method_2 should be reverted to original (x * 3, NOT the modified x * 8)
        assert "return x * 3" in final_content, "helper_method_2 should be reverted to original"
        assert "return x * 8" not in final_content, "helper_method_2 should NOT contain the modified version"

        # helper_method_1 should remain (it's still called)
        assert "def helper_method_1(self, x):" in final_content, "helper_method_1 should still exist"

        # Test reversion
        original_helper_code = {main_file: main_file.read_text()}

        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

        # Check final file content
        final_content = main_file.read_text()

        # The entrypoint method should be optimized
        assert "result1 + n * 3" in final_content, "Entrypoint method should be optimized"

        # helper_method_2 should be reverted to original
        assert "x * 3" in final_content, "helper_method_2 should still exist"

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_class_method_calls_external_helper_functions():
    """Test when class method calls external helper functions."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file with class method that calls external helpers
        main_file = temp_dir / "main.py"
        main_file.write_text("""
def external_helper_1(x):
    \"\"\"External helper function.\"\"\"
    return x * 2

def external_helper_2(x):
    \"\"\"External helper function.\"\"\"
    return x * 3

class Processor:
    def process_data(self, n):
        \"\"\"Method that calls external helper functions.\"\"\"
        result1 = external_helper_1(n)
        result2 = external_helper_2(n)
        return result1 + result2
""")

        # Optimized version that only calls one external helper
        optimized_code = f"""
{get_code_block_splitter("main.py") }
def external_helper_1(x):
    \"\"\"External helper function.\"\"\"
    return x * 2

def external_helper_2(x):
    \"\"\"External helper function - should be reverted.\"\"\"
    return x * 3

class Processor:
    def process_data(self, n):
        \"\"\"Optimized method that only calls one helper.\"\"\"
        result1 = external_helper_1(n)
        return result1 + n * 3  # Inlined external_helper_2
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Create FunctionToOptimize instance for class method
        from codeflash.models.models import FunctionParent

        function_to_optimize = FunctionToOptimize(
            file_path=main_file,
            function_name="process_data",
            qualified_name="Processor.process_data",
            parents=[FunctionParent(name="Processor", type="ClassDef")],
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection
        unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

        # Should detect external_helper_2 as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}
        expected_unused = {"external_helper_2"}

        assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # Update optimized code to include a MODIFIED unused helper
        optimized_code_with_modified_helper = f"""
{get_code_block_splitter("main.py")}
def external_helper_1(x):
    \"\"\"External helper function.\"\"\"
    return x * 2

def external_helper_2(x):
    \"\"\"External helper function - MODIFIED VERSION should be reverted.\"\"\"
    return x * 11  # This should be reverted to x * 3

class Processor:
    def process_data(self, n):
        \"\"\"Optimized method that only calls one helper.\"\"\"
        result1 = external_helper_1(n)
        return result1 + n * 3  # Inlined external_helper_2
"""

        original_helper_code = {main_file: main_file.read_text()}

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(
            code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code_with_modified_helper), original_helper_code
        )

        # Check final file content
        final_content = main_file.read_text()

        # The class method should be optimized
        assert "result1 + n * 3" in final_content, "Process method should be optimized"

        # external_helper_2 should be reverted to original (x * 3, NOT the modified x * 11)
        assert "return x * 3" in final_content, "external_helper_2 should be reverted to original"
        assert "return x * 11" not in final_content, "external_helper_2 should NOT contain the modified version"

        # external_helper_1 should remain (it's still called)
        assert "def external_helper_1(x):" in final_content, "external_helper_1 should still exist"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # Update optimized code to include a MODIFIED unused helper
        optimized_code_with_modified_helper = f"""
{get_code_block_splitter("main.py")}
def external_helper_1(x):
    \"\"\"External helper function.\"\"\"
    return x * 2

def external_helper_2(x):
    \"\"\"External helper function - MODIFIED VERSION should be reverted.\"\"\"
    return x * 7  # This should be reverted to x * 3

class Processor:
    def process_data(self, n):
        \"\"\"Optimized method that only calls one helper.\"\"\"
        result1 = external_helper_1(n)
        return result1 + n * 3  # Inlined external_helper_2
"""

        original_helper_code = {main_file: main_file.read_text()}

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(
            code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code_with_modified_helper), original_helper_code
        )

        # Check final file content
        final_content = main_file.read_text()

        # The class method should be optimized
        assert "result1 + n * 3" in final_content, "Process method should be optimized"

        # external_helper_2 should be reverted to original (x * 3, NOT the modified x * 7)
        assert "return x * 3" in final_content, "external_helper_2 should be reverted to original"
        assert "return x * 7" not in final_content, "external_helper_2 should NOT contain the modified version"

        # external_helper_1 should remain (it's still called)
        assert "def external_helper_1(x):" in final_content, "external_helper_1 should still exist"

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_nested_class_method_optimization():
    """Test optimization of methods in nested classes."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file with nested class
        main_file = temp_dir / "main.py"
        main_file.write_text("""
def global_helper_1(x):
    return x * 2

def global_helper_2(x):
    return x * 3

class OuterClass:
    class InnerProcessor:
        def compute(self, n):
            \"\"\"Method that calls global helper functions.\"\"\"
            result1 = global_helper_1(n)
            result2 = global_helper_2(n)
            return result1 + result2
            
        def local_helper(self, x):
            return x + 1
""")

        # Optimized version that inlines one helper
        optimized_code = f"""
{get_code_block_splitter("main.py") }
def global_helper_1(x):
    return x * 2

def global_helper_2(x):
    return x * 3

class OuterClass:
    class InnerProcessor:
        def compute(self, n):
            \"\"\"Optimized method.\"\"\"
            result1 = global_helper_1(n)
            return result1 + n * 3  # Inlined global_helper_2
            
        def local_helper(self, x):
            return x + 1
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Note: In practice, codeflash might not handle deeply nested classes,
        # but we test the detection logic anyway
        from codeflash.models.models import FunctionParent

        function_to_optimize = FunctionToOptimize(
            file_path=main_file,
            function_name="compute",
            qualified_name="OuterClass.InnerProcessor.compute",
            parents=[
                FunctionParent(name="OuterClass", type="ClassDef"),
                FunctionParent(name="InnerProcessor", type="ClassDef"),
            ],
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Test detection directly (context extraction might not work for nested classes)
        unused_helpers = detect_unused_helper_functions(
            optimizer.function_to_optimize,
            # Create a minimal context for testing
            type(
                "MockContext",
                (),
                {
                    "helper_functions": [
                        type(
                            "MockHelper",
                            (),
                            {
                                "qualified_name": "global_helper_1",
                                "only_function_name": "global_helper_1",
                                "fully_qualified_name": "main.global_helper_1",
                                "file_path": main_file,
                                "jedi_definition": type("MockJedi", (), {"type": "function"})(),
                            },
                        )(),
                        type(
                            "MockHelper",
                            (),
                            {
                                "qualified_name": "global_helper_2",
                                "only_function_name": "global_helper_2",
                                "fully_qualified_name": "main.global_helper_2",
                                "file_path": main_file,
                                "jedi_definition": type("MockJedi", (), {"type": "function"})(),
                            },
                        )(),
                    ]
                },
            )(),
            optimized_code,
        )

        # Should detect global_helper_2 as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}
        expected_unused = {"global_helper_2"}

        assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

        # For nested class tests, we'll skip the complete workflow test since nested classes
        # may not be fully supported by the optimizer, but we've verified detection works

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # Since this test uses nested classes which might not be fully supported,
        # we'll only test with the mock context for detection but skip the full workflow test
        # The other tests cover the complete workflow comprehensively

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_multi_file_import_styles():
    """Test detection with different import styles in multi-file projects."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file
        main_file = temp_dir / "main.py"
        main_file.write_text("""
import utils
from math_helpers import add, multiply
from processors import process_data as pd

def entrypoint_function(n):
    \"\"\"Function using different import styles.\"\"\"
    result1 = utils.compute(n)  # Module.function style
    result2 = add(n, 5)  # Direct import style
    result3 = multiply(n, 2)  # Direct import style
    result4 = pd(n)  # Aliased import style
    return result1 + result2 + result3 + result4
""")

        # Utils file
        utils_file = temp_dir / "utils.py"
        utils_file.write_text("""
def compute(x):
    \"\"\"Utility compute function.\"\"\"
    return x * 10

def unused_util(x):
    \"\"\"This utility function should be unused.\"\"\"
    return x + 100
""")

        # Math helpers file
        math_file = temp_dir / "math_helpers.py"
        math_file.write_text("""
def add(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply(x, y):
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

def subtract(x, y):
    \"\"\"Subtract function - should be unused.\"\"\"
    return x - y
""")

        # Processors file
        processors_file = temp_dir / "processors.py"
        processors_file.write_text("""
def process_data(x):
    \"\"\"Process data.\"\"\"
    return x ** 2

def clean_data(x):
    \"\"\"Clean data - should be unused.\"\"\"
    return x
""")

        # Optimized version that only uses some functions
        optimized_code = f"""
{get_code_block_splitter("main.py") }
import utils
from math_helpers import add

def entrypoint_function(n):
    \"\"\"Optimized function using fewer helpers.\"\"\"
    result1 = utils.compute(n)  # Still using utils.compute
    result2 = add(n, 5)  # Still using add
    # Inlined multiply: result3 = n * 2
    # Inlined process_data: result4 = n ** 2
    return result1 + result2 + (n * 2) + (n ** 2)
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Create FunctionToOptimize instance
        function_to_optimize = FunctionToOptimize(
            file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection
        unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

        # Should detect multiply, process_data as unused (at minimum)
        unused_names = {uh.qualified_name for uh in unused_helpers}

        # The exact unused functions may vary based on what helpers are discovered by Jedi
        # At minimum, we expect multiply to be detected as unused since it's not imported
        assert "multiply" in unused_names, "Expected multiply to be detected as unused"
        assert "process_data" in unused_names, "Expected process_data to be detected as unused"
        assert "subtract" not in unused_names, "Expected subtract not to be detected as unused"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # First modify some helper files to simulate optimization changes
        math_file.write_text("""
def add(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply(x, y):
    \"\"\"Multiply two numbers - MODIFIED VERSION.\"\"\"
    return x * y * 2  # This should be reverted to x * y

def subtract(x, y):
    \"\"\"Subtract function - should be unused.\"\"\"
    return x - y
""")

        # Store original helper code
        original_helper_code = {
            main_file: """
import utils
from math_helpers import add, multiply
from processors import process_data as pd

def entrypoint_function(n):
    \"\"\"Function using different import styles.\"\"\"
    result1 = utils.compute(n)  # Module.function style
    result2 = add(n, 5)  # Direct import style
    result3 = multiply(n, 2)  # Direct import style
    result4 = pd(n)  # Aliased import style
    return result1 + result2 + result3 + result4
""",
            utils_file: utils_file.read_text(),
            math_file: """
def add(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply(x, y):
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

def subtract(x, y):
    \"\"\"Subtract function - should be unused.\"\"\"
    return x - y
""",
            processors_file: processors_file.read_text(),
        }

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

        # Check main file content
        main_content = main_file.read_text()
        assert "(n * 2) + (n ** 2)" in main_content, "Entrypoint function should be optimized with inlined calculations"
        assert "from math_helpers import add" in main_content, (
            "Imports should be updated to only include used functions"
        )

        # Verify that unused helper files are reverted if they contained unused functions that were modified
        math_content = math_file.read_text()
        assert "def add(x, y):" in math_content, "add function should still exist"
        # If multiply was unused and modified, it should be reverted
        if "multiply" in unused_names:
            assert "return x * y" in math_content, "multiply should be reverted to original if it was unused"
            assert "return x * y * 2" not in math_content, (
                "multiply should NOT contain the modified version if it was unused"
            )

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_module_dot_function_import_style():
    """Test detection when helpers are called via module.function style."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file
        main_file = temp_dir / "main.py"
        main_file.write_text("""
import calculator

def entrypoint_function(n):
    \"\"\"Function using module.function import style.\"\"\"
    result1 = calculator.add_numbers(n, 10)
    result2 = calculator.multiply_numbers(n, 5)
    return result1 + result2
""")

        # Calculator file
        calc_file = temp_dir / "calculator.py"
        calc_file.write_text("""
def add_numbers(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply_numbers(x, y):
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

def divide_numbers(x, y):
    \"\"\"Divide function - should be unused.\"\"\"
    return x / y
""")

        # Optimized version that only uses add_numbers
        optimized_code = f"""
{get_code_block_splitter("main.py") }
import calculator

def entrypoint_function(n):
    \"\"\"Optimized function that inlines multiply.\"\"\"
    result1 = calculator.add_numbers(n, 10)
    # Inlined: result2 = n * 5
    return result1 + (n * 5)
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Create FunctionToOptimize instance
        function_to_optimize = FunctionToOptimize(
            file_path=main_file, function_name="entrypoint_function", qualified_name="entrypoint_function", parents=[]
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection
        unused_helpers = detect_unused_helper_functions(optimizer.function_to_optimize, code_context, optimized_code)

        # Should detect multiply_numbers and divide_numbers as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}

        # Check that multiply_numbers is detected as unused
        assert "multiply_numbers" in unused_names, f"Expected 'multiply_numbers' to be unused, got: {unused_names}"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # First modify the calculator file to simulate optimization changes
        calc_file.write_text("""
def add_numbers(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply_numbers(x, y):
    \"\"\"Multiply two numbers - MODIFIED VERSION.\"\"\"
    return x * y * 5  # This should be reverted to x * y

def divide_numbers(x, y):
    \"\"\"Divide function - should be unused.\"\"\"
    return x / y
""")

        # Store original helper code
        original_helper_code = {
            main_file: """
import calculator

def entrypoint_function(n):
    \"\"\"Function using module.function import style.\"\"\"
    result1 = calculator.add_numbers(n, 10)
    result2 = calculator.multiply_numbers(n, 5)
    return result1 + result2
""",
            calc_file: """
def add_numbers(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply_numbers(x, y):
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

def divide_numbers(x, y):
    \"\"\"Divide function - should be unused.\"\"\"
    return x / y
""",
        }

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

        # Check main file content
        main_content = main_file.read_text()
        assert "+ (n * 5)" in main_content, "Entrypoint function should be optimized with inlined multiplication"
        assert "import calculator" in main_content, "Calculator import should remain"

        # Check calculator file content - unused functions should be reverted if modified
        calc_content = calc_file.read_text()
        assert "def add_numbers(x, y):" in calc_content, "add_numbers should still exist"
        assert "def multiply_numbers(x, y):" in calc_content, "multiply_numbers should exist"
        assert "def divide_numbers(x, y):" in calc_content, "divide_numbers should remain as original"
        # multiply_numbers should be reverted to original since it's unused
        assert "return x * y" in calc_content, "multiply_numbers should be reverted to original"
        assert "return x * y * 5" not in calc_content, "multiply_numbers should NOT contain the modified version"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # First modify the calculator file to simulate optimization changes
        calc_file.write_text("""
def add_numbers(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply_numbers(x, y):
    \"\"\"Multiply two numbers - MODIFIED VERSION.\"\"\"
    return x * y * 3  # This should be reverted to x * y

def divide_numbers(x, y):
    \"\"\"Divide function - should be unused.\"\"\"
    return x / y
""")

        # Store original helper code
        original_helper_code = {
            main_file: """
import calculator

def entrypoint_function(n):
    \"\"\"Function using module.function import style.\"\"\"
    result1 = calculator.add_numbers(n, 10)
    result2 = calculator.multiply_numbers(n, 5)
    return result1 + result2
""",
            calc_file: """
def add_numbers(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def multiply_numbers(x, y):
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

def divide_numbers(x, y):
    \"\"\"Divide function - should be unused.\"\"\"
    return x / y
""",
        }

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(code_context, CodeStringsMarkdown.parse_flattened_code(optimized_code), original_helper_code)

        # Check main file content
        main_content = main_file.read_text()
        assert "+ (n * 5)" in main_content, "Entrypoint function should be optimized with inlined multiplication"
        assert "import calculator" in main_content, "Calculator import should remain"

        # Check calculator file content - unused functions should be reverted if modified
        calc_content = calc_file.read_text()
        assert "def add_numbers(x, y):" in calc_content, "add_numbers should still exist"
        assert "def multiply_numbers(x, y):" in calc_content, "multiply_numbers should exist"
        assert "def divide_numbers(x, y):" in calc_content, "divide_numbers should remain as original"
        # multiply_numbers should be reverted to original since it's unused
        assert "return x * y" in calc_content, "multiply_numbers should be reverted to original"
        assert "return x * y * 3" not in calc_content, "multiply_numbers should NOT contain the modified version"

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def test_static_method_and_class_method():
    """Test optimization of static methods and class methods."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Main file with static and class methods
        main_file = temp_dir / "main.py"
        main_file.write_text("""
def utility_function_1(x):
    return x * 2

def utility_function_2(x):
    return x * 3

class MathUtils:
    @staticmethod
    def calculate_static(n):
        \"\"\"Static method that calls utility functions.\"\"\"
        result1 = utility_function_1(n)
        result2 = utility_function_2(n)
        return result1 + result2
    
    @classmethod
    def calculate_class(cls, n):
        \"\"\"Class method that calls utility functions.\"\"\"
        result1 = utility_function_1(n)
        result2 = utility_function_2(n)
        return result1 - result2
""")

        # Optimized static method that inlines one utility
        optimized_static_code = f"""
{get_code_block_splitter("main.py")}
def utility_function_1(x):
    return x * 2

def utility_function_2(x):
    return x * 3

class MathUtils:
    @staticmethod
    def calculate_static(n):
        \"\"\"Optimized static method.\"\"\"
        result1 = utility_function_1(n)
        return result1 + n * 3  # Inlined utility_function_2
    
    @classmethod
    def calculate_class(cls, n):
        \"\"\"Class method that calls utility functions.\"\"\"
        result1 = utility_function_1(n)
        result2 = utility_function_2(n)
        return result1 - result2
"""

        # Create test config
        test_cfg = TestConfig(
            tests_root=temp_dir / "tests",
            tests_project_rootdir=temp_dir,
            project_root_path=temp_dir,
            test_framework="pytest",
            pytest_cmd="pytest",
        )

        # Test static method optimization
        from codeflash.models.models import FunctionParent

        function_to_optimize = FunctionToOptimize(
            file_path=main_file,
            function_name="calculate_static",
            qualified_name="MathUtils.calculate_static",
            parents=[FunctionParent(name="MathUtils", type="ClassDef")],
        )

        # Create function optimizer
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            function_to_optimize_source_code=main_file.read_text(),
        )

        # Get original code context
        ctx_result = optimizer.get_code_optimization_context()
        assert ctx_result.is_successful(), f"Failed to get context: {ctx_result.failure()}"

        code_context = ctx_result.unwrap()

        # Test unused helper detection for static method
        unused_helpers = detect_unused_helper_functions(
            optimizer.function_to_optimize, code_context, optimized_static_code
        )

        # Should detect utility_function_2 as unused
        unused_names = {uh.qualified_name for uh in unused_helpers}
        expected_unused = {"utility_function_2"}

        assert unused_names == expected_unused, f"Expected unused: {expected_unused}, got: {unused_names}"

        # Also test the complete replace_function_and_helpers_with_optimized_code workflow
        # Update optimized code to include a MODIFIED unused helper
        optimized_static_code_with_modified_helper = f"""
{get_code_block_splitter("main.py")}
def utility_function_1(x):
    return x * 2

def utility_function_2(x):
    return x * 6  # MODIFIED VERSION - should be reverted to x * 3

class MathUtils:
    @staticmethod
    def calculate_static(n):
        \"\"\"Optimized static method.\"\"\"
        result1 = utility_function_1(n)
        return result1 + n * 3  # Inlined utility_function_2
    
    @classmethod
    def calculate_class(cls, n):
        \"\"\"Class method that calls utility functions.\"\"\"
        result1 = utility_function_1(n)
        result2 = utility_function_2(n)
        return result1 - result2
"""

        original_helper_code = {main_file: main_file.read_text()}

        # Apply optimization and test reversion
        optimizer.replace_function_and_helpers_with_optimized_code(
            code_context, CodeStringsMarkdown.parse_flattened_code(optimized_static_code_with_modified_helper), original_helper_code
        )

        # Check final file content
        final_content = main_file.read_text()

        # The static method should be optimized
        assert "result1 + n * 3" in final_content, "Static method should be optimized"

        # utility_function_2 should be reverted to original (x * 3, NOT the modified x * 6)
        assert "return x * 3" in final_content, "utility_function_2 should be reverted to original"
        assert "return x * 6" not in final_content, "utility_function_2 should NOT contain the modified version"

        # utility_function_1 should remain (it's still called)
        assert "def utility_function_1(x):" in final_content, "utility_function_1 should still exist"

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
