from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00357_0():
    input_content = "4\n20\n5\n10\n1"
    expected_output = "no"
    run_pie_test_case("../p00357.py", input_content, expected_output)


def test_problem_p00357_1():
    input_content = "4\n20\n30\n1\n20"
    expected_output = "yes"
    run_pie_test_case("../p00357.py", input_content, expected_output)


def test_problem_p00357_2():
    input_content = "4\n20\n5\n10\n1"
    expected_output = "no"
    run_pie_test_case("../p00357.py", input_content, expected_output)


def test_problem_p00357_3():
    input_content = "3\n10\n5\n10"
    expected_output = "no"
    run_pie_test_case("../p00357.py", input_content, expected_output)
