from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03065_0():
    input_content = "2\n7\n-7\n14"
    expected_output = "2\n7"
    run_pie_test_case("../p03065.py", input_content, expected_output)


def test_problem_p03065_1():
    input_content = "2\n7\n-7\n14"
    expected_output = "2\n7"
    run_pie_test_case("../p03065.py", input_content, expected_output)


def test_problem_p03065_2():
    input_content = "0\n998244353"
    expected_output = "998244353"
    run_pie_test_case("../p03065.py", input_content, expected_output)
