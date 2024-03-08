from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03309_0():
    input_content = "5\n2 2 3 5 5"
    expected_output = "2"
    run_pie_test_case("../p03309.py", input_content, expected_output)


def test_problem_p03309_1():
    input_content = "5\n2 2 3 5 5"
    expected_output = "2"
    run_pie_test_case("../p03309.py", input_content, expected_output)


def test_problem_p03309_2():
    input_content = "7\n1 1 1 1 2 3 4"
    expected_output = "6"
    run_pie_test_case("../p03309.py", input_content, expected_output)


def test_problem_p03309_3():
    input_content = "9\n1 2 3 4 5 6 7 8 9"
    expected_output = "0"
    run_pie_test_case("../p03309.py", input_content, expected_output)


def test_problem_p03309_4():
    input_content = "6\n6 5 4 3 2 1"
    expected_output = "18"
    run_pie_test_case("../p03309.py", input_content, expected_output)
