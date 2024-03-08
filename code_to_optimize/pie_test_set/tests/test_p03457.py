from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03457_0():
    input_content = "2\n3 1 2\n6 1 1"
    expected_output = "Yes"
    run_pie_test_case("../p03457.py", input_content, expected_output)


def test_problem_p03457_1():
    input_content = "2\n3 1 2\n6 1 1"
    expected_output = "Yes"
    run_pie_test_case("../p03457.py", input_content, expected_output)


def test_problem_p03457_2():
    input_content = "2\n5 1 1\n100 1 1"
    expected_output = "No"
    run_pie_test_case("../p03457.py", input_content, expected_output)


def test_problem_p03457_3():
    input_content = "1\n2 100 100"
    expected_output = "No"
    run_pie_test_case("../p03457.py", input_content, expected_output)
