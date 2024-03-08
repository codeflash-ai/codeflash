from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00836_0():
    input_content = "2\n3\n17\n41\n20\n666\n12\n53\n0"
    expected_output = "1\n1\n2\n3\n0\n0\n1\n2"
    run_pie_test_case("../p00836.py", input_content, expected_output)


def test_problem_p00836_1():
    input_content = "2\n3\n17\n41\n20\n666\n12\n53\n0"
    expected_output = "1\n1\n2\n3\n0\n0\n1\n2"
    run_pie_test_case("../p00836.py", input_content, expected_output)
