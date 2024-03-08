from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02726_0():
    input_content = "5 2 4"
    expected_output = "5\n4\n1\n0"
    run_pie_test_case("../p02726.py", input_content, expected_output)


def test_problem_p02726_1():
    input_content = "7 3 7"
    expected_output = "7\n8\n4\n2\n0\n0"
    run_pie_test_case("../p02726.py", input_content, expected_output)


def test_problem_p02726_2():
    input_content = "10 4 8"
    expected_output = "10\n12\n10\n8\n4\n1\n0\n0\n0"
    run_pie_test_case("../p02726.py", input_content, expected_output)


def test_problem_p02726_3():
    input_content = "3 1 3"
    expected_output = "3\n0"
    run_pie_test_case("../p02726.py", input_content, expected_output)


def test_problem_p02726_4():
    input_content = "5 2 4"
    expected_output = "5\n4\n1\n0"
    run_pie_test_case("../p02726.py", input_content, expected_output)
