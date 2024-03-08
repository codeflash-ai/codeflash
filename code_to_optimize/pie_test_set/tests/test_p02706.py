from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02706_0():
    input_content = "41 2\n5 6"
    expected_output = "30"
    run_pie_test_case("../p02706.py", input_content, expected_output)


def test_problem_p02706_1():
    input_content = "10 2\n5 6"
    expected_output = "-1"
    run_pie_test_case("../p02706.py", input_content, expected_output)


def test_problem_p02706_2():
    input_content = "314 15\n9 26 5 35 8 9 79 3 23 8 46 2 6 43 3"
    expected_output = "9"
    run_pie_test_case("../p02706.py", input_content, expected_output)


def test_problem_p02706_3():
    input_content = "41 2\n5 6"
    expected_output = "30"
    run_pie_test_case("../p02706.py", input_content, expected_output)


def test_problem_p02706_4():
    input_content = "11 2\n5 6"
    expected_output = "0"
    run_pie_test_case("../p02706.py", input_content, expected_output)
