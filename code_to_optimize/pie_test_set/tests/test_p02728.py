from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02728_0():
    input_content = "3\n1 2\n1 3"
    expected_output = "2\n1\n1"
    run_pie_test_case("../p02728.py", input_content, expected_output)


def test_problem_p02728_1():
    input_content = "3\n1 2\n1 3"
    expected_output = "2\n1\n1"
    run_pie_test_case("../p02728.py", input_content, expected_output)


def test_problem_p02728_2():
    input_content = "2\n1 2"
    expected_output = "1\n1"
    run_pie_test_case("../p02728.py", input_content, expected_output)


def test_problem_p02728_3():
    input_content = "5\n1 2\n2 3\n3 4\n3 5"
    expected_output = "2\n8\n12\n3\n3"
    run_pie_test_case("../p02728.py", input_content, expected_output)


def test_problem_p02728_4():
    input_content = "8\n1 2\n2 3\n3 4\n3 5\n3 6\n6 7\n6 8"
    expected_output = "40\n280\n840\n120\n120\n504\n72\n72"
    run_pie_test_case("../p02728.py", input_content, expected_output)
