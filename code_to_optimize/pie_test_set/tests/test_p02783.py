from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02783_0():
    input_content = "10 4"
    expected_output = "3"
    run_pie_test_case("../p02783.py", input_content, expected_output)


def test_problem_p02783_1():
    input_content = "10 4"
    expected_output = "3"
    run_pie_test_case("../p02783.py", input_content, expected_output)


def test_problem_p02783_2():
    input_content = "1 10000"
    expected_output = "1"
    run_pie_test_case("../p02783.py", input_content, expected_output)


def test_problem_p02783_3():
    input_content = "10000 1"
    expected_output = "10000"
    run_pie_test_case("../p02783.py", input_content, expected_output)
