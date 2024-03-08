from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02785_0():
    input_content = "3 1\n4 1 5"
    expected_output = "5"
    run_pie_test_case("../p02785.py", input_content, expected_output)


def test_problem_p02785_1():
    input_content = "3 0\n1000000000 1000000000 1000000000"
    expected_output = "3000000000"
    run_pie_test_case("../p02785.py", input_content, expected_output)


def test_problem_p02785_2():
    input_content = "8 9\n7 9 3 2 3 8 4 6"
    expected_output = "0"
    run_pie_test_case("../p02785.py", input_content, expected_output)


def test_problem_p02785_3():
    input_content = "3 1\n4 1 5"
    expected_output = "5"
    run_pie_test_case("../p02785.py", input_content, expected_output)
