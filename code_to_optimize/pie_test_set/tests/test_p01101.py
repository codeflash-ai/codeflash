from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01101_0():
    input_content = "3 45\n10 20 30\n6 10\n1 2 5 8 9 11\n7 100\n11 34 83 47 59 29 70\n4 100\n80 70 60 50\n4 20\n10 5 10 16\n0 0"
    expected_output = "40\n10\n99\nNONE\n20"
    run_pie_test_case("../p01101.py", input_content, expected_output)


def test_problem_p01101_1():
    input_content = "3 45\n10 20 30\n6 10\n1 2 5 8 9 11\n7 100\n11 34 83 47 59 29 70\n4 100\n80 70 60 50\n4 20\n10 5 10 16\n0 0"
    expected_output = "40\n10\n99\nNONE\n20"
    run_pie_test_case("../p01101.py", input_content, expected_output)
