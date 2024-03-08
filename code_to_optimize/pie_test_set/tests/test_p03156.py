from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03156_0():
    input_content = "7\n5 15\n1 10 16 2 7 20 12"
    expected_output = "2"
    run_pie_test_case("../p03156.py", input_content, expected_output)


def test_problem_p03156_1():
    input_content = "3\n5 6\n5 6 10"
    expected_output = "1"
    run_pie_test_case("../p03156.py", input_content, expected_output)


def test_problem_p03156_2():
    input_content = "7\n5 15\n1 10 16 2 7 20 12"
    expected_output = "2"
    run_pie_test_case("../p03156.py", input_content, expected_output)


def test_problem_p03156_3():
    input_content = "8\n3 8\n5 5 5 10 10 10 15 20"
    expected_output = "0"
    run_pie_test_case("../p03156.py", input_content, expected_output)
