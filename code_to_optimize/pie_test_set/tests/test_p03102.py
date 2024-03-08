from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03102_0():
    input_content = "2 3 -10\n1 2 3\n3 2 1\n1 2 2"
    expected_output = "1"
    run_pie_test_case("../p03102.py", input_content, expected_output)


def test_problem_p03102_1():
    input_content = "3 3 0\n100 -100 0\n0 100 100\n100 100 100\n-100 100 100"
    expected_output = "0"
    run_pie_test_case("../p03102.py", input_content, expected_output)


def test_problem_p03102_2():
    input_content = "5 2 -4\n-2 5\n100 41\n100 40\n-3 0\n-6 -2\n18 -13"
    expected_output = "2"
    run_pie_test_case("../p03102.py", input_content, expected_output)


def test_problem_p03102_3():
    input_content = "2 3 -10\n1 2 3\n3 2 1\n1 2 2"
    expected_output = "1"
    run_pie_test_case("../p03102.py", input_content, expected_output)
