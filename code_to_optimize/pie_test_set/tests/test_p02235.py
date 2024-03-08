from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02235_0():
    input_content = "3\nabcbdab\nbdcaba\nabc\nabc\nabc\nbc"
    expected_output = "4\n3\n2"
    run_pie_test_case("../p02235.py", input_content, expected_output)


def test_problem_p02235_1():
    input_content = "3\nabcbdab\nbdcaba\nabc\nabc\nabc\nbc"
    expected_output = "4\n3\n2"
    run_pie_test_case("../p02235.py", input_content, expected_output)
