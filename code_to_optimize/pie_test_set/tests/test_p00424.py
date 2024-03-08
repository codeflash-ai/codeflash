from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00424_0():
    input_content = "3\nA a\n0 5\n5 4\n10\nA\nB\nC\n0\n1\n4\n5\na\nb\nA\n3\nA a\n0 5\n5 4\n10\nA\nB\nC\n0\n1\n4\n5\na\nb\nA\n0"
    expected_output = "aBC5144aba\naBC5144aba"
    run_pie_test_case("../p00424.py", input_content, expected_output)


def test_problem_p00424_1():
    input_content = "3\nA a\n0 5\n5 4\n10\nA\nB\nC\n0\n1\n4\n5\na\nb\nA\n3\nA a\n0 5\n5 4\n10\nA\nB\nC\n0\n1\n4\n5\na\nb\nA\n0"
    expected_output = "aBC5144aba\naBC5144aba"
    run_pie_test_case("../p00424.py", input_content, expected_output)
