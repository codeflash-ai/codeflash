from unittest.mock import Mock

from code_to_optimize.final_test_set.belongs_to_class import belongs_to_class


def test_positive_match():
    name = Mock()
    name.full_name = "module.ClassName.method"
    name.module_name = "module"

    assert belongs_to_class(
        name, "ClassName"
    ), "Should return True for correct class name"


def test_negative_match():
    name = Mock()
    name.full_name = "module.OtherClass.method"
    name.module_name = "module"

    assert not belongs_to_class(
        name, "ClassName"
    ), "Should return False for incorrect class name"


def test_incorrect_module_name():
    name = Mock()
    name.full_name = "othermodule.ClassName.method"
    name.module_name = "module"

    assert not belongs_to_class(
        name, "ClassName"
    ), "Should return False for incorrect module name"


def test_no_full_name():
    name = Mock()
    name.full_name = None  # Or an empty string ""
    name.module_name = "module"

    assert not belongs_to_class(
        name, "ClassName"
    ), "Should return False when there is no full name"


def test_class_name_as_substring():
    name = Mock()
    name.full_name = "module.ClassNameExtra.method"
    name.module_name = "module"

    assert not belongs_to_class(
        name, "ClassName"
    ), "Should return False when class name is a substring"
