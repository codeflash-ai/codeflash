from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03377_0():
    input_content = "3 5 4"
    expected_output = "YES"
    run_pie_test_case("../p03377.py", input_content, expected_output)


def test_problem_p03377_1():
    input_content = "2 2 6"
    expected_output = "NO"
    run_pie_test_case("../p03377.py", input_content, expected_output)


def test_problem_p03377_2():
    input_content = "3 5 4"
    expected_output = "YES"
    run_pie_test_case("../p03377.py", input_content, expected_output)


def test_problem_p03377_3():
    input_content = "5 3 2"
    expected_output = "NO"
    run_pie_test_case("../p03377.py", input_content, expected_output)
