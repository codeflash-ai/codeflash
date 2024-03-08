from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03831_0():
    input_content = "4 2 5\n1 2 5 7"
    expected_output = "11"
    run_pie_test_case("../p03831.py", input_content, expected_output)
