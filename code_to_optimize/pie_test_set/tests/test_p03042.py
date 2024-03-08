from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03042_0():
    input_content = "1905"
    expected_output = "YYMM"
    run_pie_test_case("../p03042.py", input_content, expected_output)


def test_problem_p03042_1():
    input_content = "1905"
    expected_output = "YYMM"
    run_pie_test_case("../p03042.py", input_content, expected_output)


def test_problem_p03042_2():
    input_content = "1700"
    expected_output = "NA"
    run_pie_test_case("../p03042.py", input_content, expected_output)


def test_problem_p03042_3():
    input_content = "0112"
    expected_output = "AMBIGUOUS"
    run_pie_test_case("../p03042.py", input_content, expected_output)
