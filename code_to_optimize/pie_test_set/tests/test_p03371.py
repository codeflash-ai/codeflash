from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03371_0():
    input_content = "1500 2000 1600 3 2"
    expected_output = "7900"
    run_pie_test_case("../p03371.py", input_content, expected_output)


def test_problem_p03371_1():
    input_content = "1500 2000 1600 3 2"
    expected_output = "7900"
    run_pie_test_case("../p03371.py", input_content, expected_output)


def test_problem_p03371_2():
    input_content = "1500 2000 500 90000 100000"
    expected_output = "100000000"
    run_pie_test_case("../p03371.py", input_content, expected_output)


def test_problem_p03371_3():
    input_content = "1500 2000 1900 3 2"
    expected_output = "8500"
    run_pie_test_case("../p03371.py", input_content, expected_output)
