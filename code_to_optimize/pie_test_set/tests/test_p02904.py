from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02904_0():
    input_content = "5 3\n0 2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p02904.py", input_content, expected_output)


def test_problem_p02904_1():
    input_content = "10 4\n2 0 1 3 7 5 4 6 8 9"
    expected_output = "6"
    run_pie_test_case("../p02904.py", input_content, expected_output)


def test_problem_p02904_2():
    input_content = "5 3\n0 2 1 4 3"
    expected_output = "2"
    run_pie_test_case("../p02904.py", input_content, expected_output)


def test_problem_p02904_3():
    input_content = "4 4\n0 1 2 3"
    expected_output = "1"
    run_pie_test_case("../p02904.py", input_content, expected_output)
