from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02793_0():
    input_content = "3\n2 3 4"
    expected_output = "13"
    run_pie_test_case("../p02793.py", input_content, expected_output)


def test_problem_p02793_1():
    input_content = "3\n2 3 4"
    expected_output = "13"
    run_pie_test_case("../p02793.py", input_content, expected_output)


def test_problem_p02793_2():
    input_content = "3\n1000000 999999 999998"
    expected_output = "996989508"
    run_pie_test_case("../p02793.py", input_content, expected_output)


def test_problem_p02793_3():
    input_content = "5\n12 12 12 12 12"
    expected_output = "5"
    run_pie_test_case("../p02793.py", input_content, expected_output)
