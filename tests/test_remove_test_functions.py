from codeflash.languages.python.support import PythonSupport


def test_remove_bare_function():
    src = """
def test_foo():
    pass

def test_bar():
    pass

def test_baz():
    pass
"""
    result = PythonSupport().remove_test_functions(src, ["test_bar"])
    assert result == """
def test_foo():
    pass

def test_baz():
    pass
"""


def test_remove_qualified_method():
    src = """
class TestSuite:
    def test_alpha(self):
        pass

    def test_beta(self):
        pass

    def test_gamma(self):
        pass
"""
    result = PythonSupport().remove_test_functions(src, ["TestSuite.test_beta"])
    assert result == """
class TestSuite:
    def test_alpha(self):
        pass

    def test_gamma(self):
        pass
"""


def test_remove_all_methods_removes_class():
    src = """
class TestSuite:
    def test_alpha(self):
        pass

    def test_beta(self):
        pass
"""
    result = PythonSupport().remove_test_functions(
        src, ["TestSuite.test_alpha", "TestSuite.test_beta"]
    )
    assert result == "\n"


def test_remove_all_methods_from_class_with_docstring():
    src = """
class TestSuite:
    \"\"\"Suite docstring.\"\"\"
    def test_only(self):
        pass
"""
    result = PythonSupport().remove_test_functions(src, ["TestSuite.test_only"])
    assert result == "\n"


def test_mixed_bare_and_qualified():
    src = """
def test_standalone():
    pass

class TestSuite:
    def test_method(self):
        pass
"""
    result = PythonSupport().remove_test_functions(
        src, ["test_standalone", "TestSuite.test_method"]
    )
    assert result == "\n"


def test_bare_name_does_not_match_class_method():
    src = """
class TestSuite:
    def test_method(self):
        pass

def test_method():
    pass
"""
    result = PythonSupport().remove_test_functions(src, ["test_method"])
    assert result == """
class TestSuite:
    def test_method(self):
        pass
"""


def test_class_kept_when_non_test_methods_remain():
    src = """
class TestSuite:
    def setUp(self):
        self.x = 1

    def test_alpha(self):
        pass

    def test_beta(self):
        pass
"""
    result = PythonSupport().remove_test_functions(
        src, ["TestSuite.test_alpha", "TestSuite.test_beta"]
    )
    assert result == """
class TestSuite:
    def setUp(self):
        self.x = 1
"""


def test_qualified_name_wrong_class_no_removal():
    src = """
class TestA:
    def test_method(self):
        pass

class TestB:
    def test_method(self):
        pass
"""
    result = PythonSupport().remove_test_functions(src, ["TestA.test_method"])
    assert result == """

class TestB:
    def test_method(self):
        pass
"""


def test_no_functions_to_remove_returns_unchanged():
    src = """
def test_foo():
    pass
"""
    result = PythonSupport().remove_test_functions(src, [])
    assert result == """
def test_foo():
    pass
"""


def test_invalid_syntax_returns_original():
    src = "def test_foo(:\n    pass"
    result = PythonSupport().remove_test_functions(src, ["test_foo"])
    assert result == src
