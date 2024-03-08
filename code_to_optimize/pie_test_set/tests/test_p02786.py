from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02786_0():
    input_content = "2"
    expected_output = "3"
    run_pie_test_case("../p02786.py", input_content, expected_output)


def test_problem_p02786_1():
    input_content = "4"
    expected_output = "7"
    run_pie_test_case("../p02786.py", input_content, expected_output)


def test_problem_p02786_2():
    input_content = "1000000000000"
    expected_output = "1099511627775"
    run_pie_test_case("../p02786.py", input_content, expected_output)


def test_problem_p02786_3():
    input_content = "2"
    expected_output = "3"
    run_pie_test_case("../p02786.py", input_content, expected_output)
