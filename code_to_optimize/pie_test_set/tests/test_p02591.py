from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02591_0():
    input_content = "3\n2 3 1 4"
    expected_output = "121788"
    run_pie_test_case("../p02591.py", input_content, expected_output)


def test_problem_p02591_1():
    input_content = "2\n1 2"
    expected_output = "36"
    run_pie_test_case("../p02591.py", input_content, expected_output)


def test_problem_p02591_2():
    input_content = "3\n2 3 1 4"
    expected_output = "121788"
    run_pie_test_case("../p02591.py", input_content, expected_output)


def test_problem_p02591_3():
    input_content = "5\n6 14 15 7 12 16 5 4 11 9 3 10 8 2 13 1"
    expected_output = "10199246"
    run_pie_test_case("../p02591.py", input_content, expected_output)
