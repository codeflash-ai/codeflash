from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02736_0():
    input_content = "4\n1231"
    expected_output = "1"
    run_pie_test_case("../p02736.py", input_content, expected_output)


def test_problem_p02736_1():
    input_content = "10\n2311312312"
    expected_output = "0"
    run_pie_test_case("../p02736.py", input_content, expected_output)


def test_problem_p02736_2():
    input_content = "4\n1231"
    expected_output = "1"
    run_pie_test_case("../p02736.py", input_content, expected_output)
