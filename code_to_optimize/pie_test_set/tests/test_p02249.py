from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02249_0():
    input_content = "4 5\n00010\n00101\n00010\n00100\n3 2\n10\n01\n10"
    expected_output = "0 3\n1 2"
    run_pie_test_case("../p02249.py", input_content, expected_output)


def test_problem_p02249_1():
    input_content = "4 5\n00010\n00101\n00010\n00100\n3 2\n10\n01\n10"
    expected_output = "0 3\n1 2"
    run_pie_test_case("../p02249.py", input_content, expected_output)
