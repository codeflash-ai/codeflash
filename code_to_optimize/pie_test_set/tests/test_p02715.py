from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02715_0():
    input_content = "3 2"
    expected_output = "9"
    run_pie_test_case("../p02715.py", input_content, expected_output)


def test_problem_p02715_1():
    input_content = "100000 100000"
    expected_output = "742202979"
    run_pie_test_case("../p02715.py", input_content, expected_output)


def test_problem_p02715_2():
    input_content = "3 200"
    expected_output = "10813692"
    run_pie_test_case("../p02715.py", input_content, expected_output)


def test_problem_p02715_3():
    input_content = "3 2"
    expected_output = "9"
    run_pie_test_case("../p02715.py", input_content, expected_output)
