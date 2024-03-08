from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03735_0():
    input_content = "3\n1 2\n3 4\n5 6"
    expected_output = "15"
    run_pie_test_case("../p03735.py", input_content, expected_output)


def test_problem_p03735_1():
    input_content = "2\n1 1\n1000000000 1000000000"
    expected_output = "999999998000000001"
    run_pie_test_case("../p03735.py", input_content, expected_output)


def test_problem_p03735_2():
    input_content = "3\n1 2\n3 4\n5 6"
    expected_output = "15"
    run_pie_test_case("../p03735.py", input_content, expected_output)


def test_problem_p03735_3():
    input_content = "3\n1010 10\n1000 1\n20 1020"
    expected_output = "380"
    run_pie_test_case("../p03735.py", input_content, expected_output)
