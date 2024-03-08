from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02789_0():
    input_content = "3 3"
    expected_output = "Yes"
    run_pie_test_case("../p02789.py", input_content, expected_output)


def test_problem_p02789_1():
    input_content = "3 3"
    expected_output = "Yes"
    run_pie_test_case("../p02789.py", input_content, expected_output)


def test_problem_p02789_2():
    input_content = "1 1"
    expected_output = "Yes"
    run_pie_test_case("../p02789.py", input_content, expected_output)


def test_problem_p02789_3():
    input_content = "3 2"
    expected_output = "No"
    run_pie_test_case("../p02789.py", input_content, expected_output)
