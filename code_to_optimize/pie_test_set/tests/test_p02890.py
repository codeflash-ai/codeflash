from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02890_0():
    input_content = "3\n2 1 2"
    expected_output = "3\n1\n0"
    run_pie_test_case("../p02890.py", input_content, expected_output)


def test_problem_p02890_1():
    input_content = "5\n1 2 3 4 5"
    expected_output = "5\n2\n1\n1\n1"
    run_pie_test_case("../p02890.py", input_content, expected_output)


def test_problem_p02890_2():
    input_content = "4\n1 3 3 3"
    expected_output = "4\n1\n0\n0"
    run_pie_test_case("../p02890.py", input_content, expected_output)


def test_problem_p02890_3():
    input_content = "3\n2 1 2"
    expected_output = "3\n1\n0"
    run_pie_test_case("../p02890.py", input_content, expected_output)
