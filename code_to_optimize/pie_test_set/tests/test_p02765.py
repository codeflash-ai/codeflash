from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02765_0():
    input_content = "2 2919"
    expected_output = "3719"
    run_pie_test_case("../p02765.py", input_content, expected_output)


def test_problem_p02765_1():
    input_content = "2 2919"
    expected_output = "3719"
    run_pie_test_case("../p02765.py", input_content, expected_output)


def test_problem_p02765_2():
    input_content = "22 3051"
    expected_output = "3051"
    run_pie_test_case("../p02765.py", input_content, expected_output)
