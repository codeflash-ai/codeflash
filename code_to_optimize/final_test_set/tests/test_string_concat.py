from code_to_optimize.final_test_set.string_concat import concatenate_strings


def test_concatenate_strings_zero():
    assert concatenate_strings(0) == "", "Failed: Expected an empty string for input 0"


def test_concatenate_strings_positive():
    assert (
        concatenate_strings(5) == "0, 1, 2, 3, 4, "
    ), "Failed: Incorrect string for input 5"


def test_concatenate_strings_large_number():
    result = concatenate_strings(1000)
    expected_length = sum(
        len(str(i)) + 2 for i in range(1000)
    )  # Each number i + len(", ")
    assert (
        len(result) == expected_length
    ), f"Failed: Incorrect length for large input 1000"
