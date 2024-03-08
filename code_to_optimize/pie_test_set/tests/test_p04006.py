from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04006_0():
    input_content = "2 10\n1 100"
    expected_output = "12"
    run_pie_test_case("../p04006.py", input_content, expected_output)


def test_problem_p04006_1():
    input_content = "4 10\n1 2 3 4"
    expected_output = "10"
    run_pie_test_case("../p04006.py", input_content, expected_output)


def test_problem_p04006_2():
    input_content = "2 10\n1 100"
    expected_output = "12"
    run_pie_test_case("../p04006.py", input_content, expected_output)


def test_problem_p04006_3():
    input_content = "3 10\n100 1 100"
    expected_output = "23"
    run_pie_test_case("../p04006.py", input_content, expected_output)
