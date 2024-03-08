from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03158_0():
    input_content = "5 5\n3 5 7 11 13\n1\n4\n9\n10\n13"
    expected_output = "31\n31\n27\n23\n23"
    run_pie_test_case("../p03158.py", input_content, expected_output)


def test_problem_p03158_1():
    input_content = "5 5\n3 5 7 11 13\n1\n4\n9\n10\n13"
    expected_output = "31\n31\n27\n23\n23"
    run_pie_test_case("../p03158.py", input_content, expected_output)


def test_problem_p03158_2():
    input_content = "4 3\n10 20 30 40\n2\n34\n34"
    expected_output = "70\n60\n60"
    run_pie_test_case("../p03158.py", input_content, expected_output)
