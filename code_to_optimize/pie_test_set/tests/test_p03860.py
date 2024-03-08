from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03860_0():
    input_content = "AtCoder Beginner Contest"
    expected_output = "ABC"
    run_pie_test_case("../p03860.py", input_content, expected_output)


def test_problem_p03860_1():
    input_content = "AtCoder Beginner Contest"
    expected_output = "ABC"
    run_pie_test_case("../p03860.py", input_content, expected_output)


def test_problem_p03860_2():
    input_content = "AtCoder Snuke Contest"
    expected_output = "ASC"
    run_pie_test_case("../p03860.py", input_content, expected_output)


def test_problem_p03860_3():
    input_content = "AtCoder X Contest"
    expected_output = "AXC"
    run_pie_test_case("../p03860.py", input_content, expected_output)
