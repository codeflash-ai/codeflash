from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00031_0():
    input_content = "5\n7\n127"
    expected_output = "1 4\n1 2 4\n1 2 4 8 16 32 64"
    run_pie_test_case("../p00031.py", input_content, expected_output)


def test_problem_p00031_1():
    input_content = "5\n7\n127"
    expected_output = "1 4\n1 2 4\n1 2 4 8 16 32 64"
    run_pie_test_case("../p00031.py", input_content, expected_output)
