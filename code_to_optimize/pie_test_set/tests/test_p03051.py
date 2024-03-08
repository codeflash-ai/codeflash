from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03051_0():
    input_content = "3\n1 2 3"
    expected_output = "3"
    run_pie_test_case("../p03051.py", input_content, expected_output)


def test_problem_p03051_1():
    input_content = "32\n0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    expected_output = "147483634"
    run_pie_test_case("../p03051.py", input_content, expected_output)


def test_problem_p03051_2():
    input_content = "3\n1 2 2"
    expected_output = "1"
    run_pie_test_case("../p03051.py", input_content, expected_output)


def test_problem_p03051_3():
    input_content = "3\n1 2 3"
    expected_output = "3"
    run_pie_test_case("../p03051.py", input_content, expected_output)


def test_problem_p03051_4():
    input_content = "24\n1 2 5 3 3 6 1 1 8 8 0 3 3 4 6 6 4 0 7 2 5 4 6 2"
    expected_output = "292"
    run_pie_test_case("../p03051.py", input_content, expected_output)
