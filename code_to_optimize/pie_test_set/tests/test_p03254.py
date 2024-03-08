from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03254_0():
    input_content = "3 70\n20 30 10"
    expected_output = "2"
    run_pie_test_case("../p03254.py", input_content, expected_output)


def test_problem_p03254_1():
    input_content = "4 1111\n1 10 100 1000"
    expected_output = "4"
    run_pie_test_case("../p03254.py", input_content, expected_output)


def test_problem_p03254_2():
    input_content = "3 70\n20 30 10"
    expected_output = "2"
    run_pie_test_case("../p03254.py", input_content, expected_output)


def test_problem_p03254_3():
    input_content = "3 10\n20 30 10"
    expected_output = "1"
    run_pie_test_case("../p03254.py", input_content, expected_output)


def test_problem_p03254_4():
    input_content = "2 10\n20 20"
    expected_output = "0"
    run_pie_test_case("../p03254.py", input_content, expected_output)
