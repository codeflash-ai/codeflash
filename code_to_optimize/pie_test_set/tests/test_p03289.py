from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03289_0():
    input_content = "AtCoder"
    expected_output = "AC"
    run_pie_test_case("../p03289.py", input_content, expected_output)


def test_problem_p03289_1():
    input_content = "AtCoCo"
    expected_output = "WA"
    run_pie_test_case("../p03289.py", input_content, expected_output)


def test_problem_p03289_2():
    input_content = "Atcoder"
    expected_output = "WA"
    run_pie_test_case("../p03289.py", input_content, expected_output)


def test_problem_p03289_3():
    input_content = "ACoder"
    expected_output = "WA"
    run_pie_test_case("../p03289.py", input_content, expected_output)


def test_problem_p03289_4():
    input_content = "AcycliC"
    expected_output = "WA"
    run_pie_test_case("../p03289.py", input_content, expected_output)


def test_problem_p03289_5():
    input_content = "AtCoder"
    expected_output = "AC"
    run_pie_test_case("../p03289.py", input_content, expected_output)
