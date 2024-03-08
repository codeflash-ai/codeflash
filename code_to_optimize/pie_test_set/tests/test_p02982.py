from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02982_0():
    input_content = "3 2\n1 2\n5 5\n-2 8"
    expected_output = "1"
    run_pie_test_case("../p02982.py", input_content, expected_output)


def test_problem_p02982_1():
    input_content = "3 4\n-3 7 8 2\n-12 1 10 2\n-2 8 9 3"
    expected_output = "2"
    run_pie_test_case("../p02982.py", input_content, expected_output)


def test_problem_p02982_2():
    input_content = "3 2\n1 2\n5 5\n-2 8"
    expected_output = "1"
    run_pie_test_case("../p02982.py", input_content, expected_output)


def test_problem_p02982_3():
    input_content = "5 1\n1\n2\n3\n4\n5"
    expected_output = "10"
    run_pie_test_case("../p02982.py", input_content, expected_output)
