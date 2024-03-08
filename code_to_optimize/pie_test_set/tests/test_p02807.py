from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02807_0():
    input_content = "3\n1 2 3"
    expected_output = "5"
    run_pie_test_case("../p02807.py", input_content, expected_output)


def test_problem_p02807_1():
    input_content = "12\n161735902 211047202 430302156 450968417 628894325 707723857 731963982 822804784 880895728 923078537 971407775 982631932"
    expected_output = "750927044"
    run_pie_test_case("../p02807.py", input_content, expected_output)


def test_problem_p02807_2():
    input_content = "3\n1 2 3"
    expected_output = "5"
    run_pie_test_case("../p02807.py", input_content, expected_output)
