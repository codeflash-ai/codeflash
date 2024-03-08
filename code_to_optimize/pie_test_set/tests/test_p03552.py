from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03552_0():
    input_content = "3 100 100\n10 1000 100"
    expected_output = "900"
    run_pie_test_case("../p03552.py", input_content, expected_output)


def test_problem_p03552_1():
    input_content = "5 1 1\n1 1 1 1 1"
    expected_output = "0"
    run_pie_test_case("../p03552.py", input_content, expected_output)


def test_problem_p03552_2():
    input_content = "3 100 1000\n10 100 100"
    expected_output = "900"
    run_pie_test_case("../p03552.py", input_content, expected_output)


def test_problem_p03552_3():
    input_content = "3 100 100\n10 1000 100"
    expected_output = "900"
    run_pie_test_case("../p03552.py", input_content, expected_output)


def test_problem_p03552_4():
    input_content = "1 1 1\n1000000000"
    expected_output = "999999999"
    run_pie_test_case("../p03552.py", input_content, expected_output)
