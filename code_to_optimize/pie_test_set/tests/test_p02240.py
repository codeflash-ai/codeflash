from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02240_0():
    input_content = "10 9\n0 1\n0 2\n3 4\n5 7\n5 6\n6 7\n6 8\n7 8\n8 9\n3\n0 1\n5 9\n1 3"
    expected_output = "yes\nyes\nno"
    run_pie_test_case("../p02240.py", input_content, expected_output)


def test_problem_p02240_1():
    input_content = "10 9\n0 1\n0 2\n3 4\n5 7\n5 6\n6 7\n6 8\n7 8\n8 9\n3\n0 1\n5 9\n1 3"
    expected_output = "yes\nyes\nno"
    run_pie_test_case("../p02240.py", input_content, expected_output)
