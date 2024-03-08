from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03472_0():
    input_content = "1 10\n3 5"
    expected_output = "3"
    run_pie_test_case("../p03472.py", input_content, expected_output)


def test_problem_p03472_1():
    input_content = "4 1000000000\n1 1\n1 10000000\n1 30000000\n1 99999999"
    expected_output = "860000004"
    run_pie_test_case("../p03472.py", input_content, expected_output)


def test_problem_p03472_2():
    input_content = "1 10\n3 5"
    expected_output = "3"
    run_pie_test_case("../p03472.py", input_content, expected_output)


def test_problem_p03472_3():
    input_content = "2 10\n3 5\n2 6"
    expected_output = "2"
    run_pie_test_case("../p03472.py", input_content, expected_output)


def test_problem_p03472_4():
    input_content = "5 500\n35 44\n28 83\n46 62\n31 79\n40 43"
    expected_output = "9"
    run_pie_test_case("../p03472.py", input_content, expected_output)
