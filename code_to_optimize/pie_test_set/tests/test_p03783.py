from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03783_0():
    input_content = "3\n1 3\n5 7\n1 3"
    expected_output = "2"
    run_pie_test_case("../p03783.py", input_content, expected_output)


def test_problem_p03783_1():
    input_content = "3\n2 5\n4 6\n1 4"
    expected_output = "0"
    run_pie_test_case("../p03783.py", input_content, expected_output)


def test_problem_p03783_2():
    input_content = "5\n123456 789012\n123 456\n12 345678901\n123456 789012\n1 23"
    expected_output = "246433"
    run_pie_test_case("../p03783.py", input_content, expected_output)


def test_problem_p03783_3():
    input_content = "3\n1 3\n5 7\n1 3"
    expected_output = "2"
    run_pie_test_case("../p03783.py", input_content, expected_output)


def test_problem_p03783_4():
    input_content = "1\n1 400"
    expected_output = "0"
    run_pie_test_case("../p03783.py", input_content, expected_output)


def test_problem_p03783_5():
    input_content = "5\n999999999 1000000000\n1 2\n314 315\n500000 500001\n999999999 1000000000"
    expected_output = "1999999680"
    run_pie_test_case("../p03783.py", input_content, expected_output)
