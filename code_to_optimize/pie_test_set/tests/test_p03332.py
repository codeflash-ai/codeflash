from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03332_0():
    input_content = "4 1 2 5"
    expected_output = "40"
    run_pie_test_case("../p03332.py", input_content, expected_output)


def test_problem_p03332_1():
    input_content = "90081 33447 90629 6391049189"
    expected_output = "577742975"
    run_pie_test_case("../p03332.py", input_content, expected_output)


def test_problem_p03332_2():
    input_content = "2 5 6 0"
    expected_output = "1"
    run_pie_test_case("../p03332.py", input_content, expected_output)


def test_problem_p03332_3():
    input_content = "4 1 2 5"
    expected_output = "40"
    run_pie_test_case("../p03332.py", input_content, expected_output)
