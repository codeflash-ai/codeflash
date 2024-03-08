from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00314_0():
    input_content = "7\n5 4 3 10 2 4 1"
    expected_output = "4"
    run_pie_test_case("../p00314.py", input_content, expected_output)


def test_problem_p00314_1():
    input_content = "7\n5 4 3 10 2 4 1"
    expected_output = "4"
    run_pie_test_case("../p00314.py", input_content, expected_output)


def test_problem_p00314_2():
    input_content = "3\n1 1 100"
    expected_output = "1"
    run_pie_test_case("../p00314.py", input_content, expected_output)


def test_problem_p00314_3():
    input_content = "4\n11 15 58 1"
    expected_output = "3"
    run_pie_test_case("../p00314.py", input_content, expected_output)
