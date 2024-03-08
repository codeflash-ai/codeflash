from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00208_0():
    input_content = "15\n100\n1000000000\n3\n0"
    expected_output = "19\n155\n9358757000\n3"
    run_pie_test_case("../p00208.py", input_content, expected_output)


def test_problem_p00208_1():
    input_content = "15\n100\n1000000000\n3\n0"
    expected_output = "19\n155\n9358757000\n3"
    run_pie_test_case("../p00208.py", input_content, expected_output)
