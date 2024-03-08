from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02421_0():
    input_content = "3\ncat dog\nfish fish\nlion tiger"
    expected_output = "1 7"
    run_pie_test_case("../p02421.py", input_content, expected_output)


def test_problem_p02421_1():
    input_content = "3\ncat dog\nfish fish\nlion tiger"
    expected_output = "1 7"
    run_pie_test_case("../p02421.py", input_content, expected_output)
