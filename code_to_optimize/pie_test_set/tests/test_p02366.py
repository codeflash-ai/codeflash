from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02366_0():
    input_content = "4 4\n0 1\n0 2\n1 2\n2 3"
    expected_output = "2"
    run_pie_test_case("../p02366.py", input_content, expected_output)


def test_problem_p02366_1():
    input_content = "5 4\n0 1\n1 2\n2 3\n3 4"
    expected_output = "1\n2\n3"
    run_pie_test_case("../p02366.py", input_content, expected_output)


def test_problem_p02366_2():
    input_content = "4 4\n0 1\n0 2\n1 2\n2 3"
    expected_output = "2"
    run_pie_test_case("../p02366.py", input_content, expected_output)
