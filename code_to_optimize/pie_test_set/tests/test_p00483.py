from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00483_0():
    input_content = "4 7\n4\nJIOJOIJ\nIOJOIJO\nJOIJOOI\nOOJJIJO\n3 5 4 7\n2 2 3 6\n2 2 2 2\n1 1 4 7"
    expected_output = "1 3 2\n3 5 2\n0 1 0\n10 11 7"
    run_pie_test_case("../p00483.py", input_content, expected_output)


def test_problem_p00483_1():
    input_content = "4 7\n4\nJIOJOIJ\nIOJOIJO\nJOIJOOI\nOOJJIJO\n3 5 4 7\n2 2 3 6\n2 2 2 2\n1 1 4 7"
    expected_output = "1 3 2\n3 5 2\n0 1 0\n10 11 7"
    run_pie_test_case("../p00483.py", input_content, expected_output)
