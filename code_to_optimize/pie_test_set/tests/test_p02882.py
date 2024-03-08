from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02882_0():
    input_content = "2 2 4"
    expected_output = "45.0000000000"
    run_pie_test_case("../p02882.py", input_content, expected_output)


def test_problem_p02882_1():
    input_content = "2 2 4"
    expected_output = "45.0000000000"
    run_pie_test_case("../p02882.py", input_content, expected_output)


def test_problem_p02882_2():
    input_content = "12 21 10"
    expected_output = "89.7834636934"
    run_pie_test_case("../p02882.py", input_content, expected_output)


def test_problem_p02882_3():
    input_content = "3 1 8"
    expected_output = "4.2363947991"
    run_pie_test_case("../p02882.py", input_content, expected_output)
