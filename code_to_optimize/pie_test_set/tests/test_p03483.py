from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03483_0():
    input_content = "eel"
    expected_output = "1"
    run_pie_test_case("../p03483.py", input_content, expected_output)


def test_problem_p03483_1():
    input_content = "snuke"
    expected_output = "-1"
    run_pie_test_case("../p03483.py", input_content, expected_output)


def test_problem_p03483_2():
    input_content = "ataatmma"
    expected_output = "4"
    run_pie_test_case("../p03483.py", input_content, expected_output)


def test_problem_p03483_3():
    input_content = "eel"
    expected_output = "1"
    run_pie_test_case("../p03483.py", input_content, expected_output)
