from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03185_0():
    input_content = "5 6\n1 2 3 4 5"
    expected_output = "20"
    run_pie_test_case("../p03185.py", input_content, expected_output)


def test_problem_p03185_1():
    input_content = "8 5\n1 3 4 5 10 11 12 13"
    expected_output = "62"
    run_pie_test_case("../p03185.py", input_content, expected_output)


def test_problem_p03185_2():
    input_content = "2 1000000000000\n500000 1000000"
    expected_output = "1250000000000"
    run_pie_test_case("../p03185.py", input_content, expected_output)


def test_problem_p03185_3():
    input_content = "5 6\n1 2 3 4 5"
    expected_output = "20"
    run_pie_test_case("../p03185.py", input_content, expected_output)
