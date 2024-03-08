from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03776_0():
    input_content = "5 2 2\n1 2 3 4 5"
    expected_output = "4.500000\n1"
    run_pie_test_case("../p03776.py", input_content, expected_output)


def test_problem_p03776_1():
    input_content = "4 2 3\n10 20 10 10"
    expected_output = "15.000000\n3"
    run_pie_test_case("../p03776.py", input_content, expected_output)


def test_problem_p03776_2():
    input_content = (
        "5 1 5\n1000000000000000 999999999999999 999999999999998 999999999999997 999999999999996"
    )
    expected_output = "1000000000000000.000000\n1"
    run_pie_test_case("../p03776.py", input_content, expected_output)


def test_problem_p03776_3():
    input_content = "5 2 2\n1 2 3 4 5"
    expected_output = "4.500000\n1"
    run_pie_test_case("../p03776.py", input_content, expected_output)
