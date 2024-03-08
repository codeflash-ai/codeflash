from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02949_0():
    input_content = "3 3 10\n1 2 20\n2 3 30\n1 3 45"
    expected_output = "35"
    run_pie_test_case("../p02949.py", input_content, expected_output)


def test_problem_p02949_1():
    input_content = "2 2 10\n1 2 100\n2 2 100"
    expected_output = "-1"
    run_pie_test_case("../p02949.py", input_content, expected_output)


def test_problem_p02949_2():
    input_content = "4 5 10\n1 2 1\n1 4 1\n3 4 1\n2 2 100\n3 3 100"
    expected_output = "0"
    run_pie_test_case("../p02949.py", input_content, expected_output)


def test_problem_p02949_3():
    input_content = "3 3 10\n1 2 20\n2 3 30\n1 3 45"
    expected_output = "35"
    run_pie_test_case("../p02949.py", input_content, expected_output)
