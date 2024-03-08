from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03178_0():
    input_content = "30\n4"
    expected_output = "6"
    run_pie_test_case("../p03178.py", input_content, expected_output)


def test_problem_p03178_1():
    input_content = "98765432109876543210\n58"
    expected_output = "635270834"
    run_pie_test_case("../p03178.py", input_content, expected_output)


def test_problem_p03178_2():
    input_content = "30\n4"
    expected_output = "6"
    run_pie_test_case("../p03178.py", input_content, expected_output)


def test_problem_p03178_3():
    input_content = "1000000009\n1"
    expected_output = "2"
    run_pie_test_case("../p03178.py", input_content, expected_output)
