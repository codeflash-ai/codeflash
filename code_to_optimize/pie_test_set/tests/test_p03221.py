from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03221_0():
    input_content = "2 3\n1 32\n2 63\n1 12"
    expected_output = "000001000002\n000002000001\n000001000001"
    run_pie_test_case("../p03221.py", input_content, expected_output)


def test_problem_p03221_1():
    input_content = "2 3\n1 32\n2 63\n1 12"
    expected_output = "000001000002\n000002000001\n000001000001"
    run_pie_test_case("../p03221.py", input_content, expected_output)


def test_problem_p03221_2():
    input_content = "2 3\n2 55\n2 77\n2 99"
    expected_output = "000002000001\n000002000002\n000002000003"
    run_pie_test_case("../p03221.py", input_content, expected_output)
