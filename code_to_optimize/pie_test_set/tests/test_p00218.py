from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00218_0():
    input_content = "4\n100 70 20\n98 86 55\n80 34 36\n65 79 65\n2\n99 81 20\n66 72 90\n0"
    expected_output = "A\nA\nB\nC\nA\nB"
    run_pie_test_case("../p00218.py", input_content, expected_output)


def test_problem_p00218_1():
    input_content = "4\n100 70 20\n98 86 55\n80 34 36\n65 79 65\n2\n99 81 20\n66 72 90\n0"
    expected_output = "A\nA\nB\nC\nA\nB"
    run_pie_test_case("../p00218.py", input_content, expected_output)
