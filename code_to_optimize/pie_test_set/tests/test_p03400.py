from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03400_0():
    input_content = "3\n7 1\n2\n5\n10"
    expected_output = "8"
    run_pie_test_case("../p03400.py", input_content, expected_output)


def test_problem_p03400_1():
    input_content = "5\n30 44\n26\n18\n81\n18\n6"
    expected_output = "56"
    run_pie_test_case("../p03400.py", input_content, expected_output)


def test_problem_p03400_2():
    input_content = "3\n7 1\n2\n5\n10"
    expected_output = "8"
    run_pie_test_case("../p03400.py", input_content, expected_output)


def test_problem_p03400_3():
    input_content = "2\n8 20\n1\n10"
    expected_output = "29"
    run_pie_test_case("../p03400.py", input_content, expected_output)
