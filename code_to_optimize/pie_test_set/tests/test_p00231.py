from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00231_0():
    input_content = "3\n80 0 30\n50 5 25\n90 27 50\n3\n80 0 30\n70 5 25\n71 30 50\n0"
    expected_output = "NG\nOK"
    run_pie_test_case("../p00231.py", input_content, expected_output)


def test_problem_p00231_1():
    input_content = "3\n80 0 30\n50 5 25\n90 27 50\n3\n80 0 30\n70 5 25\n71 30 50\n0"
    expected_output = "NG\nOK"
    run_pie_test_case("../p00231.py", input_content, expected_output)
