from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03132_0():
    input_content = "4\n1\n0\n2\n3"
    expected_output = "1"
    run_pie_test_case("../p03132.py", input_content, expected_output)


def test_problem_p03132_1():
    input_content = "4\n1\n0\n2\n3"
    expected_output = "1"
    run_pie_test_case("../p03132.py", input_content, expected_output)


def test_problem_p03132_2():
    input_content = "8\n2\n0\n0\n2\n1\n3\n4\n1"
    expected_output = "3"
    run_pie_test_case("../p03132.py", input_content, expected_output)


def test_problem_p03132_3():
    input_content = "7\n314159265\n358979323\n846264338\n327950288\n419716939\n937510582\n0"
    expected_output = "1"
    run_pie_test_case("../p03132.py", input_content, expected_output)
