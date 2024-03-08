from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03827_0():
    input_content = "5\nIIDID"
    expected_output = "2"
    run_pie_test_case("../p03827.py", input_content, expected_output)


def test_problem_p03827_1():
    input_content = "5\nIIDID"
    expected_output = "2"
    run_pie_test_case("../p03827.py", input_content, expected_output)


def test_problem_p03827_2():
    input_content = "7\nDDIDDII"
    expected_output = "0"
    run_pie_test_case("../p03827.py", input_content, expected_output)
