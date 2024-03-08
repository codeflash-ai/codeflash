from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03229_0():
    input_content = "5\n6\n8\n1\n2\n3"
    expected_output = "21"
    run_pie_test_case("../p03229.py", input_content, expected_output)


def test_problem_p03229_1():
    input_content = "3\n5\n5\n1"
    expected_output = "8"
    run_pie_test_case("../p03229.py", input_content, expected_output)


def test_problem_p03229_2():
    input_content = "6\n3\n1\n4\n1\n5\n9"
    expected_output = "25"
    run_pie_test_case("../p03229.py", input_content, expected_output)


def test_problem_p03229_3():
    input_content = "5\n6\n8\n1\n2\n3"
    expected_output = "21"
    run_pie_test_case("../p03229.py", input_content, expected_output)
