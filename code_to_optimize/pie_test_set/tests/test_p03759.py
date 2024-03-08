from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03759_0():
    input_content = "2 4 6"
    expected_output = "YES"
    run_pie_test_case("../p03759.py", input_content, expected_output)


def test_problem_p03759_1():
    input_content = "2 5 6"
    expected_output = "NO"
    run_pie_test_case("../p03759.py", input_content, expected_output)


def test_problem_p03759_2():
    input_content = "3 2 1"
    expected_output = "YES"
    run_pie_test_case("../p03759.py", input_content, expected_output)


def test_problem_p03759_3():
    input_content = "2 4 6"
    expected_output = "YES"
    run_pie_test_case("../p03759.py", input_content, expected_output)
