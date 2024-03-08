from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00078_0():
    input_content = "3\n5\n0"
    expected_output = "4   9   2\n   3   5   7\n   8   1   6\n  11  24   7  20   3\n   4  12  25   8  16\n  17   5  13  21   9\n  10  18   1  14  22\n  23   6  19   2  15"
    run_pie_test_case("../p00078.py", input_content, expected_output)


def test_problem_p00078_1():
    input_content = "3\n5\n0"
    expected_output = "4   9   2\n   3   5   7\n   8   1   6\n  11  24   7  20   3\n   4  12  25   8  16\n  17   5  13  21   9\n  10  18   1  14  22\n  23   6  19   2  15"
    run_pie_test_case("../p00078.py", input_content, expected_output)
