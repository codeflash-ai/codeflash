from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02842_0():
    input_content = "432"
    expected_output = "400"
    run_pie_test_case("../p02842.py", input_content, expected_output)


def test_problem_p02842_1():
    input_content = "1001"
    expected_output = "927"
    run_pie_test_case("../p02842.py", input_content, expected_output)


def test_problem_p02842_2():
    input_content = "1079"
    expected_output = ":("
    run_pie_test_case("../p02842.py", input_content, expected_output)


def test_problem_p02842_3():
    input_content = "432"
    expected_output = "400"
    run_pie_test_case("../p02842.py", input_content, expected_output)
