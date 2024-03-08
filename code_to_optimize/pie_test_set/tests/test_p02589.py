from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02589_0():
    input_content = "3\nabcxyx\ncyx\nabc"
    expected_output = "1"
    run_pie_test_case("../p02589.py", input_content, expected_output)


def test_problem_p02589_1():
    input_content = "6\nb\na\nabc\nc\nd\nab"
    expected_output = "5"
    run_pie_test_case("../p02589.py", input_content, expected_output)


def test_problem_p02589_2():
    input_content = "3\nabcxyx\ncyx\nabc"
    expected_output = "1"
    run_pie_test_case("../p02589.py", input_content, expected_output)
