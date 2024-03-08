from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00915_0():
    input_content = "3 6\nR 1\nL 2\nL 5\n1 10\nR 1\n2 10\nR 5\nL 7\n2 10\nR 3\nL 8\n2 99\nR 1\nL 98\n4 10\nL 1\nR 2\nL 8\nR 9\n6 10\nR 2\nR 3\nL 4\nR 6\nL 7\nL 8\n0 0"
    expected_output = "5 1\n9 1\n7 1\n8 2\n98 2\n8 2\n8 3"
    run_pie_test_case("../p00915.py", input_content, expected_output)


def test_problem_p00915_1():
    input_content = "3 6\nR 1\nL 2\nL 5\n1 10\nR 1\n2 10\nR 5\nL 7\n2 10\nR 3\nL 8\n2 99\nR 1\nL 98\n4 10\nL 1\nR 2\nL 8\nR 9\n6 10\nR 2\nR 3\nL 4\nR 6\nL 7\nL 8\n0 0"
    expected_output = "5 1\n9 1\n7 1\n8 2\n98 2\n8 2\n8 3"
    run_pie_test_case("../p00915.py", input_content, expected_output)
