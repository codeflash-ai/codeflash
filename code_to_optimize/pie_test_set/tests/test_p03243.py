from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03243_0():
    input_content = "111"
    expected_output = "111"
    run_pie_test_case("../p03243.py", input_content, expected_output)


def test_problem_p03243_1():
    input_content = "750"
    expected_output = "777"
    run_pie_test_case("../p03243.py", input_content, expected_output)


def test_problem_p03243_2():
    input_content = "112"
    expected_output = "222"
    run_pie_test_case("../p03243.py", input_content, expected_output)


def test_problem_p03243_3():
    input_content = "111"
    expected_output = "111"
    run_pie_test_case("../p03243.py", input_content, expected_output)
