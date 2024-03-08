from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03993_0():
    input_content = "4\n2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p03993.py", input_content, expected_output)


def test_problem_p03993_1():
    input_content = "3\n2 3 1"
    expected_output = "0"
    run_pie_test_case("../p03993.py", input_content, expected_output)


def test_problem_p03993_2():
    input_content = "5\n5 5 5 5 1"
    expected_output = "1"
    run_pie_test_case("../p03993.py", input_content, expected_output)


def test_problem_p03993_3():
    input_content = "4\n2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p03993.py", input_content, expected_output)
