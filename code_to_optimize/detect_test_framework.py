import ast
import os
import re
from typing import Optional


def detect_test_framework(curdir, tests_root) -> Optional[str]:
    test_framework = None
    pytest_files = ["pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"]
    pytest_config_patterns = {
        "pytest.ini": r"\[pytest\]",
        "pyproject.toml": r"\[tool\.pytest\.ini_options\]",
        "tox.ini": r"\[pytest\]",
        "setup.cfg": r"\[tool:pytest\]",
    }
    for pytest_file in pytest_files:
        file_path = os.path.join(curdir, pytest_file)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf8") as file:
                contents = file.read()
                if re.search(pytest_config_patterns[pytest_file], contents):
                    test_framework = "pytest"
                    break
        test_framework = "pytest"
    else:
        # Check if any python files contain a class that inherits from unittest.TestCase
        for filename in os.listdir(tests_root):
            if filename.endswith(".py"):
                with open(
                    os.path.join(tests_root, filename), "r", encoding="utf8"
                ) as file:
                    contents = file.read()
                    try:
                        node = ast.parse(contents)
                    except SyntaxError:
                        continue
                    if any(
                        isinstance(item, ast.ClassDef)
                        and any(
                            isinstance(base, ast.Attribute)
                            and base.attr == "TestCase"
                            or isinstance(base, ast.Name)
                            and base.id == "TestCase"
                            for base in item.bases
                        )
                        for item in node.body
                    ):
                        test_framework = "unittest"
                        break
    return test_framework
