from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00239_0():
    input_content = "4\n1 7 14 47\n2 5 35 55\n3 6 3 59\n4 6 5 15\n10 15 50 400\n2\n1 8 10 78\n2 4 18 33\n10 10 50 300\n0"
    expected_output = "1\n4\nNA"
    run_pie_test_case("../p00239.py", input_content, expected_output)


def test_problem_p00239_1():
    input_content = "4\n1 7 14 47\n2 5 35 55\n3 6 3 59\n4 6 5 15\n10 15 50 400\n2\n1 8 10 78\n2 4 18 33\n10 10 50 300\n0"
    expected_output = "1\n4\nNA"
    run_pie_test_case("../p00239.py", input_content, expected_output)
