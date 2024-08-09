import re


def remove_functions_from_generated_tests(generated_tests, test_functions_to_remove):
    for test_function in test_functions_to_remove:
        function_pattern = re.compile(
            rf"(@pytest\.mark\.parametrize\(.*?\)\s*)?def\s+{re.escape(test_function)}\(.*?\):.*?(?=\ndef\s|$)",
            re.DOTALL,
        )

        match = function_pattern.search(generated_tests.generated_original_test_source)

        if match is None or "@pytest.mark.parametrize" in match.group(0):
            continue

        generated_tests.generated_original_test_source = function_pattern.sub(
            "", generated_tests.generated_original_test_source
        )

    return generated_tests
