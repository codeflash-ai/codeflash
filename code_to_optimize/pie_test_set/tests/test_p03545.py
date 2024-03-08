from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03545_0():
    input_content = "1222"
    expected_output = "1+2+2+2=7"
    run_pie_test_case("../p03545.py", input_content, expected_output)


def test_problem_p03545_1():
    input_content = "1222"
    expected_output = "1+2+2+2=7"
    run_pie_test_case("../p03545.py", input_content, expected_output)


def test_problem_p03545_2():
    input_content = "3242"
    expected_output = "3+2+4-2=7"
    run_pie_test_case("../p03545.py", input_content, expected_output)


def test_problem_p03545_3():
    input_content = "0290"
    expected_output = "0-2+9+0=7"
    run_pie_test_case("../p03545.py", input_content, expected_output)
