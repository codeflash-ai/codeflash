from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03767_0():
    input_content = "2\n5 2 8 5 1 5"
    expected_output = "10"
    run_pie_test_case("../p03767.py", input_content, expected_output)


def test_problem_p03767_1():
    input_content = "2\n5 2 8 5 1 5"
    expected_output = "10"
    run_pie_test_case("../p03767.py", input_content, expected_output)


def test_problem_p03767_2():
    input_content = "10\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000"
    expected_output = "10000000000"
    run_pie_test_case("../p03767.py", input_content, expected_output)
