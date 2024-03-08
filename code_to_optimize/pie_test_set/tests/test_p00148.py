from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00148_0():
    input_content = "50\n5576\n5577\n5578"
    expected_output = "3C11\n3C38\n3C39\n3C01"
    run_pie_test_case("../p00148.py", input_content, expected_output)


def test_problem_p00148_1():
    input_content = "50\n5576\n5577\n5578"
    expected_output = "3C11\n3C38\n3C39\n3C01"
    run_pie_test_case("../p00148.py", input_content, expected_output)
