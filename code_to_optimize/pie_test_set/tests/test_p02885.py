from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02885_0():
    input_content = "12 4"
    expected_output = "4"
    run_pie_test_case("../p02885.py", input_content, expected_output)


def test_problem_p02885_1():
    input_content = "20 30"
    expected_output = "0"
    run_pie_test_case("../p02885.py", input_content, expected_output)


def test_problem_p02885_2():
    input_content = "12 4"
    expected_output = "4"
    run_pie_test_case("../p02885.py", input_content, expected_output)


def test_problem_p02885_3():
    input_content = "20 15"
    expected_output = "0"
    run_pie_test_case("../p02885.py", input_content, expected_output)
