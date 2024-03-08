from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03435_0():
    input_content = "1 0 1\n2 1 2\n1 0 1"
    expected_output = "Yes"
    run_pie_test_case("../p03435.py", input_content, expected_output)


def test_problem_p03435_1():
    input_content = "1 0 1\n2 1 2\n1 0 1"
    expected_output = "Yes"
    run_pie_test_case("../p03435.py", input_content, expected_output)


def test_problem_p03435_2():
    input_content = "1 8 6\n2 9 7\n0 7 7"
    expected_output = "No"
    run_pie_test_case("../p03435.py", input_content, expected_output)


def test_problem_p03435_3():
    input_content = "0 8 8\n0 8 8\n0 8 8"
    expected_output = "Yes"
    run_pie_test_case("../p03435.py", input_content, expected_output)


def test_problem_p03435_4():
    input_content = "2 2 2\n2 1 2\n2 2 2"
    expected_output = "No"
    run_pie_test_case("../p03435.py", input_content, expected_output)
