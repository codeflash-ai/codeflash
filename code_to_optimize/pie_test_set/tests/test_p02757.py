from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02757_0():
    input_content = "4 3\n3543"
    expected_output = "6"
    run_pie_test_case("../p02757.py", input_content, expected_output)


def test_problem_p02757_1():
    input_content = "4 3\n3543"
    expected_output = "6"
    run_pie_test_case("../p02757.py", input_content, expected_output)


def test_problem_p02757_2():
    input_content = "20 11\n33883322005544116655"
    expected_output = "68"
    run_pie_test_case("../p02757.py", input_content, expected_output)


def test_problem_p02757_3():
    input_content = "4 2\n2020"
    expected_output = "10"
    run_pie_test_case("../p02757.py", input_content, expected_output)
