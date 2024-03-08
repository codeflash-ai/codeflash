from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02616_0():
    input_content = "4 2\n1 2 -3 -4"
    expected_output = "12"
    run_pie_test_case("../p02616.py", input_content, expected_output)


def test_problem_p02616_1():
    input_content = "4 2\n1 2 -3 -4"
    expected_output = "12"
    run_pie_test_case("../p02616.py", input_content, expected_output)


def test_problem_p02616_2():
    input_content = "10 10\n1000000000 100000000 10000000 1000000 100000 10000 1000 100 10 1"
    expected_output = "999983200"
    run_pie_test_case("../p02616.py", input_content, expected_output)


def test_problem_p02616_3():
    input_content = "4 3\n-1 -2 -3 -4"
    expected_output = "1000000001"
    run_pie_test_case("../p02616.py", input_content, expected_output)


def test_problem_p02616_4():
    input_content = "2 1\n-1 1000000000"
    expected_output = "1000000000"
    run_pie_test_case("../p02616.py", input_content, expected_output)
