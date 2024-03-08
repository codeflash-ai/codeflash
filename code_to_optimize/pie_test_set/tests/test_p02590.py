from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02590_0():
    input_content = "4\n2019 0 2020 200002"
    expected_output = "474287"
    run_pie_test_case("../p02590.py", input_content, expected_output)


def test_problem_p02590_1():
    input_content = "4\n2019 0 2020 200002"
    expected_output = "474287"
    run_pie_test_case("../p02590.py", input_content, expected_output)


def test_problem_p02590_2():
    input_content = "5\n1 1 2 2 100000"
    expected_output = "600013"
    run_pie_test_case("../p02590.py", input_content, expected_output)
