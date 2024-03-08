from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03447_0():
    input_content = "1234\n150\n100"
    expected_output = "84"
    run_pie_test_case("../p03447.py", input_content, expected_output)


def test_problem_p03447_1():
    input_content = "1000\n108\n108"
    expected_output = "28"
    run_pie_test_case("../p03447.py", input_content, expected_output)


def test_problem_p03447_2():
    input_content = "1234\n150\n100"
    expected_output = "84"
    run_pie_test_case("../p03447.py", input_content, expected_output)


def test_problem_p03447_3():
    input_content = "579\n123\n456"
    expected_output = "0"
    run_pie_test_case("../p03447.py", input_content, expected_output)


def test_problem_p03447_4():
    input_content = "7477\n549\n593"
    expected_output = "405"
    run_pie_test_case("../p03447.py", input_content, expected_output)
