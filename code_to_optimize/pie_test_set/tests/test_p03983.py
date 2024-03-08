from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03983_0():
    input_content = "2\n20 8\n20 12"
    expected_output = "24\n20"
    run_pie_test_case("../p03983.py", input_content, expected_output)


def test_problem_p03983_1():
    input_content = "1\n20 3"
    expected_output = "67"
    run_pie_test_case("../p03983.py", input_content, expected_output)


def test_problem_p03983_2():
    input_content = "1\n200 1"
    expected_output = "148322100"
    run_pie_test_case("../p03983.py", input_content, expected_output)


def test_problem_p03983_3():
    input_content = "2\n20 8\n20 12"
    expected_output = "24\n20"
    run_pie_test_case("../p03983.py", input_content, expected_output)
