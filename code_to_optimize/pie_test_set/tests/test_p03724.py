from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03724_0():
    input_content = "4 4\n1 2\n2 4\n1 3\n3 4"
    expected_output = "YES"
    run_pie_test_case("../p03724.py", input_content, expected_output)


def test_problem_p03724_1():
    input_content = "5 5\n1 2\n3 5\n5 1\n3 4\n2 3"
    expected_output = "NO"
    run_pie_test_case("../p03724.py", input_content, expected_output)


def test_problem_p03724_2():
    input_content = "4 4\n1 2\n2 4\n1 3\n3 4"
    expected_output = "YES"
    run_pie_test_case("../p03724.py", input_content, expected_output)
