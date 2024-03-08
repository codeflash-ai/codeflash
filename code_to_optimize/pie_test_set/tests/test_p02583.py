from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02583_0():
    input_content = "5\n4 4 9 7 5"
    expected_output = "5"
    run_pie_test_case("../p02583.py", input_content, expected_output)


def test_problem_p02583_1():
    input_content = "2\n1 1"
    expected_output = "0"
    run_pie_test_case("../p02583.py", input_content, expected_output)


def test_problem_p02583_2():
    input_content = "10\n9 4 6 1 9 6 10 6 6 8"
    expected_output = "39"
    run_pie_test_case("../p02583.py", input_content, expected_output)


def test_problem_p02583_3():
    input_content = "6\n4 5 4 3 3 5"
    expected_output = "8"
    run_pie_test_case("../p02583.py", input_content, expected_output)


def test_problem_p02583_4():
    input_content = "5\n4 4 9 7 5"
    expected_output = "5"
    run_pie_test_case("../p02583.py", input_content, expected_output)
