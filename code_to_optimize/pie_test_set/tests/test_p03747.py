from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03747_0():
    input_content = "3 8 3\n0 1\n3 2\n6 1"
    expected_output = "1\n3\n0"
    run_pie_test_case("../p03747.py", input_content, expected_output)


def test_problem_p03747_1():
    input_content = "4 20 9\n7 2\n9 1\n12 1\n18 1"
    expected_output = "7\n18\n18\n1"
    run_pie_test_case("../p03747.py", input_content, expected_output)


def test_problem_p03747_2():
    input_content = "3 8 3\n0 1\n3 2\n6 1"
    expected_output = "1\n3\n0"
    run_pie_test_case("../p03747.py", input_content, expected_output)
