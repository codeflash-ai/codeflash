from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02935_0():
    input_content = "2\n3 4"
    expected_output = "3.5"
    run_pie_test_case("../p02935.py", input_content, expected_output)


def test_problem_p02935_1():
    input_content = "2\n3 4"
    expected_output = "3.5"
    run_pie_test_case("../p02935.py", input_content, expected_output)


def test_problem_p02935_2():
    input_content = "3\n500 300 200"
    expected_output = "375"
    run_pie_test_case("../p02935.py", input_content, expected_output)


def test_problem_p02935_3():
    input_content = "5\n138 138 138 138 138"
    expected_output = "138"
    run_pie_test_case("../p02935.py", input_content, expected_output)
