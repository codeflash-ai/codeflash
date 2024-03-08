from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02984_0():
    input_content = "3\n2 2 4"
    expected_output = "4 0 4"
    run_pie_test_case("../p02984.py", input_content, expected_output)


def test_problem_p02984_1():
    input_content = "3\n1000000000 1000000000 0"
    expected_output = "0 2000000000 0"
    run_pie_test_case("../p02984.py", input_content, expected_output)


def test_problem_p02984_2():
    input_content = "5\n3 8 7 5 5"
    expected_output = "2 4 12 2 8"
    run_pie_test_case("../p02984.py", input_content, expected_output)


def test_problem_p02984_3():
    input_content = "3\n2 2 4"
    expected_output = "4 0 4"
    run_pie_test_case("../p02984.py", input_content, expected_output)
