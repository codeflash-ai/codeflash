from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03987_0():
    input_content = "3\n2 1 3"
    expected_output = "9"
    run_pie_test_case("../p03987.py", input_content, expected_output)


def test_problem_p03987_1():
    input_content = "3\n2 1 3"
    expected_output = "9"
    run_pie_test_case("../p03987.py", input_content, expected_output)


def test_problem_p03987_2():
    input_content = "4\n1 3 2 4"
    expected_output = "19"
    run_pie_test_case("../p03987.py", input_content, expected_output)


def test_problem_p03987_3():
    input_content = "8\n5 4 8 1 2 6 7 3"
    expected_output = "85"
    run_pie_test_case("../p03987.py", input_content, expected_output)
