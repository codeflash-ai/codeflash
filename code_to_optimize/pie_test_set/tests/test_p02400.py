from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02400_0():
    input_content = "2"
    expected_output = "12.566371 12.566371"
    run_pie_test_case("../p02400.py", input_content, expected_output)


def test_problem_p02400_1():
    input_content = "2"
    expected_output = "12.566371 12.566371"
    run_pie_test_case("../p02400.py", input_content, expected_output)


def test_problem_p02400_2():
    input_content = "3"
    expected_output = "28.274334 18.849556"
    run_pie_test_case("../p02400.py", input_content, expected_output)
