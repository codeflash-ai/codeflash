from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02879_0():
    input_content = "2 5"
    expected_output = "10"
    run_pie_test_case("../p02879.py", input_content, expected_output)


def test_problem_p02879_1():
    input_content = "2 5"
    expected_output = "10"
    run_pie_test_case("../p02879.py", input_content, expected_output)


def test_problem_p02879_2():
    input_content = "9 9"
    expected_output = "81"
    run_pie_test_case("../p02879.py", input_content, expected_output)


def test_problem_p02879_3():
    input_content = "5 10"
    expected_output = "-1"
    run_pie_test_case("../p02879.py", input_content, expected_output)
