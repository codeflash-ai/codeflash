from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02958_0():
    input_content = "5\n5 2 3 4 1"
    expected_output = "YES"
    run_pie_test_case("../p02958.py", input_content, expected_output)


def test_problem_p02958_1():
    input_content = "5\n2 4 3 5 1"
    expected_output = "NO"
    run_pie_test_case("../p02958.py", input_content, expected_output)


def test_problem_p02958_2():
    input_content = "7\n1 2 3 4 5 6 7"
    expected_output = "YES"
    run_pie_test_case("../p02958.py", input_content, expected_output)


def test_problem_p02958_3():
    input_content = "5\n5 2 3 4 1"
    expected_output = "YES"
    run_pie_test_case("../p02958.py", input_content, expected_output)
