from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02731_0():
    input_content = "3"
    expected_output = "1.000000000000"
    run_pie_test_case("../p02731.py", input_content, expected_output)


def test_problem_p02731_1():
    input_content = "999"
    expected_output = "36926037.000000000000"
    run_pie_test_case("../p02731.py", input_content, expected_output)


def test_problem_p02731_2():
    input_content = "3"
    expected_output = "1.000000000000"
    run_pie_test_case("../p02731.py", input_content, expected_output)
