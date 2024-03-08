from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02646_0():
    input_content = "1 2\n3 1\n3"
    expected_output = "YES"
    run_pie_test_case("../p02646.py", input_content, expected_output)


def test_problem_p02646_1():
    input_content = "1 2\n3 3\n3"
    expected_output = "NO"
    run_pie_test_case("../p02646.py", input_content, expected_output)


def test_problem_p02646_2():
    input_content = "1 2\n3 2\n3"
    expected_output = "NO"
    run_pie_test_case("../p02646.py", input_content, expected_output)


def test_problem_p02646_3():
    input_content = "1 2\n3 1\n3"
    expected_output = "YES"
    run_pie_test_case("../p02646.py", input_content, expected_output)
