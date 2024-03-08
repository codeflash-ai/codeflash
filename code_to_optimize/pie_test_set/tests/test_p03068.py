from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03068_0():
    input_content = "5\nerror\n2"
    expected_output = "*rr*r"
    run_pie_test_case("../p03068.py", input_content, expected_output)


def test_problem_p03068_1():
    input_content = "6\neleven\n5"
    expected_output = "e*e*e*"
    run_pie_test_case("../p03068.py", input_content, expected_output)


def test_problem_p03068_2():
    input_content = "9\neducation\n7"
    expected_output = "******i**"
    run_pie_test_case("../p03068.py", input_content, expected_output)


def test_problem_p03068_3():
    input_content = "5\nerror\n2"
    expected_output = "*rr*r"
    run_pie_test_case("../p03068.py", input_content, expected_output)
