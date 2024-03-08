from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00150_0():
    input_content = "12\n100\n200\n300\n0"
    expected_output = "5 7\n71 73\n197 199\n281 283"
    run_pie_test_case("../p00150.py", input_content, expected_output)


def test_problem_p00150_1():
    input_content = "12\n100\n200\n300\n0"
    expected_output = "5 7\n71 73\n197 199\n281 283"
    run_pie_test_case("../p00150.py", input_content, expected_output)
