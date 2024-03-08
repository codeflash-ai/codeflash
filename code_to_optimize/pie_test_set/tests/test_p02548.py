from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02548_0():
    input_content = "3"
    expected_output = "3"
    run_pie_test_case("../p02548.py", input_content, expected_output)


def test_problem_p02548_1():
    input_content = "3"
    expected_output = "3"
    run_pie_test_case("../p02548.py", input_content, expected_output)


def test_problem_p02548_2():
    input_content = "100"
    expected_output = "473"
    run_pie_test_case("../p02548.py", input_content, expected_output)


def test_problem_p02548_3():
    input_content = "1000000"
    expected_output = "13969985"
    run_pie_test_case("../p02548.py", input_content, expected_output)
