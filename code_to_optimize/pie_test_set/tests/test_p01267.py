from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p01267_0():
    input_content = "1 5 7 11 10\n10\n2 5 7 11 10\n2 4\n2 1 1 256 0\n128 255\n2 0 0 1 0\n1234 5678\n2 1 1 100 0\n99 98\n2 1 1 100 0\n99 99\n2 1 1 10000 0\n1 0\n2 1 1 10000 0\n2 1\n0 0 0 0 0"
    expected_output = "0\n3\n255\n-1\n198\n199\n10000\n-1"
    run_pie_test_case("../p01267.py", input_content, expected_output)


def test_problem_p01267_1():
    input_content = "1 5 7 11 10\n10\n2 5 7 11 10\n2 4\n2 1 1 256 0\n128 255\n2 0 0 1 0\n1234 5678\n2 1 1 100 0\n99 98\n2 1 1 100 0\n99 99\n2 1 1 10000 0\n1 0\n2 1 1 10000 0\n2 1\n0 0 0 0 0"
    expected_output = "0\n3\n255\n-1\n198\n199\n10000\n-1"
    run_pie_test_case("../p01267.py", input_content, expected_output)
