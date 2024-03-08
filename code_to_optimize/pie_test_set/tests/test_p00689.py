from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00689_0():
    input_content = "8\n   60  70\n   20  40\n   50  50\n   70  10\n   10  70\n   80  90\n   70  50\n  100  50\n5\n   60  50\n    0  80\n   60  20\n   10  20\n   60  80\n0"
    expected_output = "388.9\n250.0"
    run_pie_test_case("../p00689.py", input_content, expected_output)
