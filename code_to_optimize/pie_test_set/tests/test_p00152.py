from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00152_0():
    input_content = "3\n1010 6 3 10 7 1 0 7 9 1 10 6 2 4 3 9 1 9 0\n1200 5 3 9 1 7 1 0 0 8 1 10 10 4 3 9 1 8 2 9\n1101 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3\n4\n3321 8 2 10 9 1 7 0 10 10 10 0 8 10 10 10 10\n3332 5 0 10 9 1 4 1 9 0 10 10 7 1 5 2 8 1\n3335 10 10 10 10 10 10 10 10 10 10 10 10\n3340 8 2 7 3 6 4 8 2 8 2 9 1 7 3 6 4 8 2 9 1 7\n0"
    expected_output = "1200 127\n1010 123\n1101 60\n3335 300\n3321 200\n3340 175\n3332 122"
    run_pie_test_case("../p00152.py", input_content, expected_output)


def test_problem_p00152_1():
    input_content = "3\n1010 6 3 10 7 1 0 7 9 1 10 6 2 4 3 9 1 9 0\n1200 5 3 9 1 7 1 0 0 8 1 10 10 4 3 9 1 8 2 9\n1101 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3\n4\n3321 8 2 10 9 1 7 0 10 10 10 0 8 10 10 10 10\n3332 5 0 10 9 1 4 1 9 0 10 10 7 1 5 2 8 1\n3335 10 10 10 10 10 10 10 10 10 10 10 10\n3340 8 2 7 3 6 4 8 2 8 2 9 1 7 3 6 4 8 2 9 1 7\n0"
    expected_output = "1200 127\n1010 123\n1101 60\n3335 300\n3321 200\n3340 175\n3332 122"
    run_pie_test_case("../p00152.py", input_content, expected_output)
