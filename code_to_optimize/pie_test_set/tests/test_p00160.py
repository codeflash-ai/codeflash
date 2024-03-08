from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00160_0():
    input_content = "2\n50 25 5 5\n80 60 10 30\n3\n10 15 25 24\n5 8 12 5\n30 30 30 18\n0"
    expected_output = "800\n3800"
    run_pie_test_case("../p00160.py", input_content, expected_output)


def test_problem_p00160_1():
    input_content = "2\n50 25 5 5\n80 60 10 30\n3\n10 15 25 24\n5 8 12 5\n30 30 30 18\n0"
    expected_output = "800\n3800"
    run_pie_test_case("../p00160.py", input_content, expected_output)
