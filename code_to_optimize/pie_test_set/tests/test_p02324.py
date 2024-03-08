from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02324_0():
    input_content = "4 4\n0 1 1\n0 2 2\n1 3 3\n2 3 4"
    expected_output = "10"
    run_pie_test_case("../p02324.py", input_content, expected_output)


def test_problem_p02324_1():
    input_content = "4 5\n0 1 1\n0 2 2\n1 3 3\n2 3 4\n1 2 5"
    expected_output = "18"
    run_pie_test_case("../p02324.py", input_content, expected_output)


def test_problem_p02324_2():
    input_content = "4 4\n0 1 1\n0 2 2\n1 3 3\n2 3 4"
    expected_output = "10"
    run_pie_test_case("../p02324.py", input_content, expected_output)


def test_problem_p02324_3():
    input_content = "2 3\n0 1 1\n0 1 2\n0 1 3"
    expected_output = "7"
    run_pie_test_case("../p02324.py", input_content, expected_output)
