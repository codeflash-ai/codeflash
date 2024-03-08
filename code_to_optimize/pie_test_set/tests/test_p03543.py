from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03543_0():
    input_content = "1118"
    expected_output = "Yes"
    run_pie_test_case("../p03543.py", input_content, expected_output)


def test_problem_p03543_1():
    input_content = "7777"
    expected_output = "Yes"
    run_pie_test_case("../p03543.py", input_content, expected_output)


def test_problem_p03543_2():
    input_content = "1118"
    expected_output = "Yes"
    run_pie_test_case("../p03543.py", input_content, expected_output)


def test_problem_p03543_3():
    input_content = "1234"
    expected_output = "No"
    run_pie_test_case("../p03543.py", input_content, expected_output)
