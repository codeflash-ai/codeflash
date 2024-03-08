from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03103_0():
    input_content = "2 5\n4 9\n2 4"
    expected_output = "12"
    run_pie_test_case("../p03103.py", input_content, expected_output)


def test_problem_p03103_1():
    input_content = "4 30\n6 18\n2 5\n3 10\n7 9"
    expected_output = "130"
    run_pie_test_case("../p03103.py", input_content, expected_output)


def test_problem_p03103_2():
    input_content = "1 100000\n1000000000 100000"
    expected_output = "100000000000000"
    run_pie_test_case("../p03103.py", input_content, expected_output)


def test_problem_p03103_3():
    input_content = "2 5\n4 9\n2 4"
    expected_output = "12"
    run_pie_test_case("../p03103.py", input_content, expected_output)
