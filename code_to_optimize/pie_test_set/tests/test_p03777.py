from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03777_0():
    input_content = "H H"
    expected_output = "H"
    run_pie_test_case("../p03777.py", input_content, expected_output)


def test_problem_p03777_1():
    input_content = "D D"
    expected_output = "H"
    run_pie_test_case("../p03777.py", input_content, expected_output)


def test_problem_p03777_2():
    input_content = "H H"
    expected_output = "H"
    run_pie_test_case("../p03777.py", input_content, expected_output)


def test_problem_p03777_3():
    input_content = "D H"
    expected_output = "D"
    run_pie_test_case("../p03777.py", input_content, expected_output)
