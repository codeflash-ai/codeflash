from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00222_0():
    input_content = "13\n14\n15\n16\n17\n18\n19\n20\n10000\n0"
    expected_output = "13\n13\n13\n13\n13\n13\n19\n19\n9439"
    run_pie_test_case("../p00222.py", input_content, expected_output)


def test_problem_p00222_1():
    input_content = "13\n14\n15\n16\n17\n18\n19\n20\n10000\n0"
    expected_output = "13\n13\n13\n13\n13\n13\n19\n19\n9439"
    run_pie_test_case("../p00222.py", input_content, expected_output)
