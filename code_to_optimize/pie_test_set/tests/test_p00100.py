from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00100_0():
    input_content = "4\n1001 2000 520\n1002 1800 450\n1003 1600 625\n1001 200 1220\n2\n1001 100 3\n1005 1000 100\n2\n2013 5000 100\n2013 5000 100\n0"
    expected_output = "1001\n1003\nNA\n2013"
    run_pie_test_case("../p00100.py", input_content, expected_output)


def test_problem_p00100_1():
    input_content = "4\n1001 2000 520\n1002 1800 450\n1003 1600 625\n1001 200 1220\n2\n1001 100 3\n1005 1000 100\n2\n2013 5000 100\n2013 5000 100\n0"
    expected_output = "1001\n1003\nNA\n2013"
    run_pie_test_case("../p00100.py", input_content, expected_output)
