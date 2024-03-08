from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03324_0():
    input_content = "0 5"
    expected_output = "5"
    run_pie_test_case("../p03324.py", input_content, expected_output)


def test_problem_p03324_1():
    input_content = "2 85"
    expected_output = "850000"
    run_pie_test_case("../p03324.py", input_content, expected_output)


def test_problem_p03324_2():
    input_content = "1 11"
    expected_output = "1100"
    run_pie_test_case("../p03324.py", input_content, expected_output)


def test_problem_p03324_3():
    input_content = "0 5"
    expected_output = "5"
    run_pie_test_case("../p03324.py", input_content, expected_output)
