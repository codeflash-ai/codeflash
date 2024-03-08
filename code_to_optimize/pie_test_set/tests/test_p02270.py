from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02270_0():
    input_content = "5 3\n8\n1\n7\n3\n9"
    expected_output = "10"
    run_pie_test_case("../p02270.py", input_content, expected_output)


def test_problem_p02270_1():
    input_content = "5 3\n8\n1\n7\n3\n9"
    expected_output = "10"
    run_pie_test_case("../p02270.py", input_content, expected_output)


def test_problem_p02270_2():
    input_content = "4 2\n1\n2\n2\n6"
    expected_output = "6"
    run_pie_test_case("../p02270.py", input_content, expected_output)
