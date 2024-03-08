from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02953_0():
    input_content = "5\n1 2 1 1 3"
    expected_output = "Yes"
    run_pie_test_case("../p02953.py", input_content, expected_output)


def test_problem_p02953_1():
    input_content = "1\n1000000000"
    expected_output = "Yes"
    run_pie_test_case("../p02953.py", input_content, expected_output)


def test_problem_p02953_2():
    input_content = "4\n1 3 2 1"
    expected_output = "No"
    run_pie_test_case("../p02953.py", input_content, expected_output)


def test_problem_p02953_3():
    input_content = "5\n1 2 3 4 5"
    expected_output = "Yes"
    run_pie_test_case("../p02953.py", input_content, expected_output)


def test_problem_p02953_4():
    input_content = "5\n1 2 1 1 3"
    expected_output = "Yes"
    run_pie_test_case("../p02953.py", input_content, expected_output)
