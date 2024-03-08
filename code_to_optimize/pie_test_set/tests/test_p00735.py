from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00735_0():
    input_content = "205920\n262144\n262200\n279936\n299998\n1"
    expected_output = "205920: 6 8 13 15 20 22 55 99\n262144: 8\n262200: 6 8 15 20 50 57 69 76 92 190 230 475 575 874 2185\n279936: 6 8 27\n299998: 299998"
    run_pie_test_case("../p00735.py", input_content, expected_output)


def test_problem_p00735_1():
    input_content = "205920\n262144\n262200\n279936\n299998\n1"
    expected_output = "205920: 6 8 13 15 20 22 55 99\n262144: 8\n262200: 6 8 15 20 50 57 69 76 92 190 230 475 575 874 2185\n279936: 6 8 27\n299998: 299998"
    run_pie_test_case("../p00735.py", input_content, expected_output)
