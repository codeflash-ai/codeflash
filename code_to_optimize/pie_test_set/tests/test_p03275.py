from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03275_0():
    input_content = "3\n10 30 20"
    expected_output = "30"
    run_pie_test_case("../p03275.py", input_content, expected_output)


def test_problem_p03275_1():
    input_content = "3\n10 30 20"
    expected_output = "30"
    run_pie_test_case("../p03275.py", input_content, expected_output)


def test_problem_p03275_2():
    input_content = "10\n5 9 5 9 8 9 3 5 4 3"
    expected_output = "8"
    run_pie_test_case("../p03275.py", input_content, expected_output)


def test_problem_p03275_3():
    input_content = "1\n10"
    expected_output = "10"
    run_pie_test_case("../p03275.py", input_content, expected_output)
