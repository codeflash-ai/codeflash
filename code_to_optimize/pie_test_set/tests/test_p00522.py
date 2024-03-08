from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00522_0():
    input_content = "4 3\n180\n160\n170\n190\n2 100\n3 120\n4 250"
    expected_output = "480"
    run_pie_test_case("../p00522.py", input_content, expected_output)
