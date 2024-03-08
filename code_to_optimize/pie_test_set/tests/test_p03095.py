from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03095_0():
    input_content = "4\nabcd"
    expected_output = "15"
    run_pie_test_case("../p03095.py", input_content, expected_output)


def test_problem_p03095_1():
    input_content = "3\nbaa"
    expected_output = "5"
    run_pie_test_case("../p03095.py", input_content, expected_output)


def test_problem_p03095_2():
    input_content = "4\nabcd"
    expected_output = "15"
    run_pie_test_case("../p03095.py", input_content, expected_output)


def test_problem_p03095_3():
    input_content = "5\nabcab"
    expected_output = "17"
    run_pie_test_case("../p03095.py", input_content, expected_output)
