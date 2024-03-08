from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04011_0():
    input_content = "5\n3\n10000\n9000"
    expected_output = "48000"
    run_pie_test_case("../p04011.py", input_content, expected_output)


def test_problem_p04011_1():
    input_content = "2\n3\n10000\n9000"
    expected_output = "20000"
    run_pie_test_case("../p04011.py", input_content, expected_output)


def test_problem_p04011_2():
    input_content = "5\n3\n10000\n9000"
    expected_output = "48000"
    run_pie_test_case("../p04011.py", input_content, expected_output)
