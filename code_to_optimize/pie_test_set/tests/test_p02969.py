from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02969_0():
    input_content = "4"
    expected_output = "48"
    run_pie_test_case("../p02969.py", input_content, expected_output)


def test_problem_p02969_1():
    input_content = "4"
    expected_output = "48"
    run_pie_test_case("../p02969.py", input_content, expected_output)


def test_problem_p02969_2():
    input_content = "15"
    expected_output = "675"
    run_pie_test_case("../p02969.py", input_content, expected_output)


def test_problem_p02969_3():
    input_content = "80"
    expected_output = "19200"
    run_pie_test_case("../p02969.py", input_content, expected_output)
