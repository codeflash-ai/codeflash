from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02792_0():
    input_content = "25"
    expected_output = "17"
    run_pie_test_case("../p02792.py", input_content, expected_output)


def test_problem_p02792_1():
    input_content = "2020"
    expected_output = "40812"
    run_pie_test_case("../p02792.py", input_content, expected_output)


def test_problem_p02792_2():
    input_content = "200000"
    expected_output = "400000008"
    run_pie_test_case("../p02792.py", input_content, expected_output)


def test_problem_p02792_3():
    input_content = "25"
    expected_output = "17"
    run_pie_test_case("../p02792.py", input_content, expected_output)


def test_problem_p02792_4():
    input_content = "100"
    expected_output = "108"
    run_pie_test_case("../p02792.py", input_content, expected_output)


def test_problem_p02792_5():
    input_content = "1"
    expected_output = "1"
    run_pie_test_case("../p02792.py", input_content, expected_output)
