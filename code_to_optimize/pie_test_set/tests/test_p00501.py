from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00501_0():
    input_content = "4\nbar\nabracadabra\nbear\nbar\nbaraxbara"
    expected_output = "3"
    run_pie_test_case("../p00501.py", input_content, expected_output)


def test_problem_p00501_1():
    input_content = "4\nbar\nabracadabra\nbear\nbar\nbaraxbara"
    expected_output = "3"
    run_pie_test_case("../p00501.py", input_content, expected_output)
