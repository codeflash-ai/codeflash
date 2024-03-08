from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03576_0():
    input_content = "4 4\n1 4\n3 3\n6 2\n8 1"
    expected_output = "21"
    run_pie_test_case("../p03576.py", input_content, expected_output)


def test_problem_p03576_1():
    input_content = "4 3\n-1000000000 -1000000000\n1000000000 1000000000\n-999999999 999999999\n999999999 -999999999"
    expected_output = "3999999996000000001"
    run_pie_test_case("../p03576.py", input_content, expected_output)


def test_problem_p03576_2():
    input_content = "4 2\n0 0\n1 1\n2 2\n3 3"
    expected_output = "1"
    run_pie_test_case("../p03576.py", input_content, expected_output)


def test_problem_p03576_3():
    input_content = "4 4\n1 4\n3 3\n6 2\n8 1"
    expected_output = "21"
    run_pie_test_case("../p03576.py", input_content, expected_output)
