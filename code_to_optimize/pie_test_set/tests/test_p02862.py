from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02862_0():
    input_content = "3 3"
    expected_output = "2"
    run_pie_test_case("../p02862.py", input_content, expected_output)


def test_problem_p02862_1():
    input_content = "999999 999999"
    expected_output = "151840682"
    run_pie_test_case("../p02862.py", input_content, expected_output)


def test_problem_p02862_2():
    input_content = "2 2"
    expected_output = "0"
    run_pie_test_case("../p02862.py", input_content, expected_output)


def test_problem_p02862_3():
    input_content = "3 3"
    expected_output = "2"
    run_pie_test_case("../p02862.py", input_content, expected_output)
