from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02361_0():
    input_content = "4 5 0\n0 1 1\n0 2 4\n1 2 2\n2 3 1\n1 3 5"
    expected_output = "0\n1\n3\n4"
    run_pie_test_case("../p02361.py", input_content, expected_output)


def test_problem_p02361_1():
    input_content = "4 6 1\n0 1 1\n0 2 4\n2 0 1\n1 2 2\n3 1 1\n3 2 5"
    expected_output = "3\n0\n2\nINF"
    run_pie_test_case("../p02361.py", input_content, expected_output)


def test_problem_p02361_2():
    input_content = "4 5 0\n0 1 1\n0 2 4\n1 2 2\n2 3 1\n1 3 5"
    expected_output = "0\n1\n3\n4"
    run_pie_test_case("../p02361.py", input_content, expected_output)
