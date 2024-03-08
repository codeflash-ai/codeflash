from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02937_0():
    input_content = "contest\nson"
    expected_output = "10"
    run_pie_test_case("../p02937.py", input_content, expected_output)


def test_problem_p02937_1():
    input_content = "contest\nsentence"
    expected_output = "33"
    run_pie_test_case("../p02937.py", input_content, expected_output)


def test_problem_p02937_2():
    input_content = "contest\nson"
    expected_output = "10"
    run_pie_test_case("../p02937.py", input_content, expected_output)


def test_problem_p02937_3():
    input_content = "contest\nprogramming"
    expected_output = "-1"
    run_pie_test_case("../p02937.py", input_content, expected_output)
