from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02246_0():
    input_content = "1 2 3 4\n6 7 8 0\n5 10 11 12\n9 13 14 15"
    expected_output = "8"
    run_pie_test_case("../p02246.py", input_content, expected_output)


def test_problem_p02246_1():
    input_content = "1 2 3 4\n6 7 8 0\n5 10 11 12\n9 13 14 15"
    expected_output = "8"
    run_pie_test_case("../p02246.py", input_content, expected_output)
