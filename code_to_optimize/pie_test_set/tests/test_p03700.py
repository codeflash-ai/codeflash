from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03700_0():
    input_content = "4 5 3\n8\n7\n4\n2"
    expected_output = "2"
    run_pie_test_case("../p03700.py", input_content, expected_output)


def test_problem_p03700_1():
    input_content = "5 2 1\n900000000\n900000000\n1000000000\n1000000000\n1000000000"
    expected_output = "800000000"
    run_pie_test_case("../p03700.py", input_content, expected_output)


def test_problem_p03700_2():
    input_content = "4 5 3\n8\n7\n4\n2"
    expected_output = "2"
    run_pie_test_case("../p03700.py", input_content, expected_output)


def test_problem_p03700_3():
    input_content = "2 10 4\n20\n20"
    expected_output = "4"
    run_pie_test_case("../p03700.py", input_content, expected_output)
