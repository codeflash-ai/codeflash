from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03719_0():
    input_content = "1 3 2"
    expected_output = "Yes"
    run_pie_test_case("../p03719.py", input_content, expected_output)


def test_problem_p03719_1():
    input_content = "6 5 4"
    expected_output = "No"
    run_pie_test_case("../p03719.py", input_content, expected_output)


def test_problem_p03719_2():
    input_content = "2 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03719.py", input_content, expected_output)


def test_problem_p03719_3():
    input_content = "1 3 2"
    expected_output = "Yes"
    run_pie_test_case("../p03719.py", input_content, expected_output)
