from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02563_0():
    input_content = "4 5\n1 2 3 4\n5 6 7 8 9"
    expected_output = "5 16 34 60 70 70 59 36"
    run_pie_test_case("../p02563.py", input_content, expected_output)


def test_problem_p02563_1():
    input_content = "4 5\n1 2 3 4\n5 6 7 8 9"
    expected_output = "5 16 34 60 70 70 59 36"
    run_pie_test_case("../p02563.py", input_content, expected_output)


def test_problem_p02563_2():
    input_content = "1 1\n10000000\n10000000"
    expected_output = "871938225"
    run_pie_test_case("../p02563.py", input_content, expected_output)
