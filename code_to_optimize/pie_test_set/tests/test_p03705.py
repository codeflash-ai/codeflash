from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03705_0():
    input_content = "4 4 6"
    expected_output = "5"
    run_pie_test_case("../p03705.py", input_content, expected_output)


def test_problem_p03705_1():
    input_content = "5 4 3"
    expected_output = "0"
    run_pie_test_case("../p03705.py", input_content, expected_output)


def test_problem_p03705_2():
    input_content = "1 7 10"
    expected_output = "0"
    run_pie_test_case("../p03705.py", input_content, expected_output)


def test_problem_p03705_3():
    input_content = "1 3 3"
    expected_output = "1"
    run_pie_test_case("../p03705.py", input_content, expected_output)


def test_problem_p03705_4():
    input_content = "4 4 6"
    expected_output = "5"
    run_pie_test_case("../p03705.py", input_content, expected_output)
