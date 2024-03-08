from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03698_0():
    input_content = "uncopyrightable"
    expected_output = "yes"
    run_pie_test_case("../p03698.py", input_content, expected_output)


def test_problem_p03698_1():
    input_content = "uncopyrightable"
    expected_output = "yes"
    run_pie_test_case("../p03698.py", input_content, expected_output)


def test_problem_p03698_2():
    input_content = "different"
    expected_output = "no"
    run_pie_test_case("../p03698.py", input_content, expected_output)


def test_problem_p03698_3():
    input_content = "no"
    expected_output = "yes"
    run_pie_test_case("../p03698.py", input_content, expected_output)
