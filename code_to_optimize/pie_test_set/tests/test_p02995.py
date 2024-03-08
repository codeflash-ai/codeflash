from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02995_0():
    input_content = "4 9 2 3"
    expected_output = "2"
    run_pie_test_case("../p02995.py", input_content, expected_output)


def test_problem_p02995_1():
    input_content = "314159265358979323 846264338327950288 419716939 937510582"
    expected_output = "532105071133627368"
    run_pie_test_case("../p02995.py", input_content, expected_output)


def test_problem_p02995_2():
    input_content = "10 40 6 8"
    expected_output = "23"
    run_pie_test_case("../p02995.py", input_content, expected_output)


def test_problem_p02995_3():
    input_content = "4 9 2 3"
    expected_output = "2"
    run_pie_test_case("../p02995.py", input_content, expected_output)
