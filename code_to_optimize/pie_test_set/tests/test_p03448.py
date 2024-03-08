from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03448_0():
    input_content = "2\n2\n2\n100"
    expected_output = "2"
    run_pie_test_case("../p03448.py", input_content, expected_output)


def test_problem_p03448_1():
    input_content = "2\n2\n2\n100"
    expected_output = "2"
    run_pie_test_case("../p03448.py", input_content, expected_output)


def test_problem_p03448_2():
    input_content = "30\n40\n50\n6000"
    expected_output = "213"
    run_pie_test_case("../p03448.py", input_content, expected_output)


def test_problem_p03448_3():
    input_content = "5\n1\n0\n150"
    expected_output = "0"
    run_pie_test_case("../p03448.py", input_content, expected_output)
