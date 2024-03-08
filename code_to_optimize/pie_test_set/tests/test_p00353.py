from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00353_0():
    input_content = "1000 3000 3000"
    expected_output = "2000"
    run_pie_test_case("../p00353.py", input_content, expected_output)


def test_problem_p00353_1():
    input_content = "5000 3000 4500"
    expected_output = "0"
    run_pie_test_case("../p00353.py", input_content, expected_output)


def test_problem_p00353_2():
    input_content = "1000 3000 3000"
    expected_output = "2000"
    run_pie_test_case("../p00353.py", input_content, expected_output)


def test_problem_p00353_3():
    input_content = "500 1000 2000"
    expected_output = "NA"
    run_pie_test_case("../p00353.py", input_content, expected_output)
