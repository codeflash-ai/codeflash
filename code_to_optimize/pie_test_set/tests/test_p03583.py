from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03583_0():
    input_content = "2"
    expected_output = "1 2 2"
    run_pie_test_case("../p03583.py", input_content, expected_output)


def test_problem_p03583_1():
    input_content = "2"
    expected_output = "1 2 2"
    run_pie_test_case("../p03583.py", input_content, expected_output)


def test_problem_p03583_2():
    input_content = "4664"
    expected_output = "3498 3498 3498"
    run_pie_test_case("../p03583.py", input_content, expected_output)


def test_problem_p03583_3():
    input_content = "3485"
    expected_output = "872 1012974 1539173474040"
    run_pie_test_case("../p03583.py", input_content, expected_output)
