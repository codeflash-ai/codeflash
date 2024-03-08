from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02555_0():
    input_content = "7"
    expected_output = "3"
    run_pie_test_case("../p02555.py", input_content, expected_output)


def test_problem_p02555_1():
    input_content = "7"
    expected_output = "3"
    run_pie_test_case("../p02555.py", input_content, expected_output)


def test_problem_p02555_2():
    input_content = "1729"
    expected_output = "294867501"
    run_pie_test_case("../p02555.py", input_content, expected_output)


def test_problem_p02555_3():
    input_content = "2"
    expected_output = "0"
    run_pie_test_case("../p02555.py", input_content, expected_output)
