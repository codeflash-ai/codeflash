from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03168_0():
    input_content = "3\n0.30 0.60 0.80"
    expected_output = "0.612"
    run_pie_test_case("../p03168.py", input_content, expected_output)


def test_problem_p03168_1():
    input_content = "5\n0.42 0.01 0.42 0.99 0.42"
    expected_output = "0.3821815872"
    run_pie_test_case("../p03168.py", input_content, expected_output)


def test_problem_p03168_2():
    input_content = "1\n0.50"
    expected_output = "0.5"
    run_pie_test_case("../p03168.py", input_content, expected_output)


def test_problem_p03168_3():
    input_content = "3\n0.30 0.60 0.80"
    expected_output = "0.612"
    run_pie_test_case("../p03168.py", input_content, expected_output)
