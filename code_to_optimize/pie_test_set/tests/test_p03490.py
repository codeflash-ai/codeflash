from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03490_0():
    input_content = "FTFFTFFF\n4 2"
    expected_output = "Yes"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_1():
    input_content = "TF\n1 0"
    expected_output = "No"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_2():
    input_content = "FF\n1 0"
    expected_output = "No"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_3():
    input_content = "FFTTFF\n0 0"
    expected_output = "Yes"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_4():
    input_content = "FTFFTFFF\n-2 -2"
    expected_output = "Yes"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_5():
    input_content = "TTTT\n1 0"
    expected_output = "No"
    run_pie_test_case("../p03490.py", input_content, expected_output)


def test_problem_p03490_6():
    input_content = "FTFFTFFF\n4 2"
    expected_output = "Yes"
    run_pie_test_case("../p03490.py", input_content, expected_output)
