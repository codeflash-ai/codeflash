from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03679_0():
    input_content = "4 3 6"
    expected_output = "safe"
    run_pie_test_case("../p03679.py", input_content, expected_output)


def test_problem_p03679_1():
    input_content = "4 3 6"
    expected_output = "safe"
    run_pie_test_case("../p03679.py", input_content, expected_output)


def test_problem_p03679_2():
    input_content = "6 5 1"
    expected_output = "delicious"
    run_pie_test_case("../p03679.py", input_content, expected_output)


def test_problem_p03679_3():
    input_content = "3 7 12"
    expected_output = "dangerous"
    run_pie_test_case("../p03679.py", input_content, expected_output)
