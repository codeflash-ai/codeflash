from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02622_0():
    input_content = "cupofcoffee\ncupofhottea"
    expected_output = "4"
    run_pie_test_case("../p02622.py", input_content, expected_output)


def test_problem_p02622_1():
    input_content = "apple\napple"
    expected_output = "0"
    run_pie_test_case("../p02622.py", input_content, expected_output)


def test_problem_p02622_2():
    input_content = "abcde\nbcdea"
    expected_output = "5"
    run_pie_test_case("../p02622.py", input_content, expected_output)


def test_problem_p02622_3():
    input_content = "cupofcoffee\ncupofhottea"
    expected_output = "4"
    run_pie_test_case("../p02622.py", input_content, expected_output)
