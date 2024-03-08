from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03025_0():
    input_content = "1 25 25 50"
    expected_output = "2"
    run_pie_test_case("../p03025.py", input_content, expected_output)


def test_problem_p03025_1():
    input_content = "4 50 50 0"
    expected_output = "312500008"
    run_pie_test_case("../p03025.py", input_content, expected_output)


def test_problem_p03025_2():
    input_content = "1 25 25 50"
    expected_output = "2"
    run_pie_test_case("../p03025.py", input_content, expected_output)


def test_problem_p03025_3():
    input_content = "1 100 0 0"
    expected_output = "1"
    run_pie_test_case("../p03025.py", input_content, expected_output)


def test_problem_p03025_4():
    input_content = "100000 31 41 28"
    expected_output = "104136146"
    run_pie_test_case("../p03025.py", input_content, expected_output)
