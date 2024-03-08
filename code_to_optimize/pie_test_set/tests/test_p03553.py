from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03553_0():
    input_content = "6\n1 2 -6 4 5 3"
    expected_output = "12"
    run_pie_test_case("../p03553.py", input_content, expected_output)


def test_problem_p03553_1():
    input_content = "6\n100 -100 -100 -100 100 -100"
    expected_output = "200"
    run_pie_test_case("../p03553.py", input_content, expected_output)


def test_problem_p03553_2():
    input_content = "6\n1 2 -6 4 5 3"
    expected_output = "12"
    run_pie_test_case("../p03553.py", input_content, expected_output)


def test_problem_p03553_3():
    input_content = "2\n-1000 100000"
    expected_output = "99000"
    run_pie_test_case("../p03553.py", input_content, expected_output)


def test_problem_p03553_4():
    input_content = "5\n-1 -2 -3 -4 -5"
    expected_output = "0"
    run_pie_test_case("../p03553.py", input_content, expected_output)
