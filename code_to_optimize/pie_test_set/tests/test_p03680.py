from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03680_0():
    input_content = "3\n3\n1\n2"
    expected_output = "2"
    run_pie_test_case("../p03680.py", input_content, expected_output)


def test_problem_p03680_1():
    input_content = "3\n3\n1\n2"
    expected_output = "2"
    run_pie_test_case("../p03680.py", input_content, expected_output)


def test_problem_p03680_2():
    input_content = "5\n3\n3\n4\n2\n4"
    expected_output = "3"
    run_pie_test_case("../p03680.py", input_content, expected_output)


def test_problem_p03680_3():
    input_content = "4\n3\n4\n1\n2"
    expected_output = "-1"
    run_pie_test_case("../p03680.py", input_content, expected_output)
