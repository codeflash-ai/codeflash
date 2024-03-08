from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00075_0():
    input_content = "1001,50.0,1.60\n1002,60.0,1.70\n1003,70.0,1.80\n1004,80.0,1.70\n1005,90.0,1.60"
    expected_output = "1004\n1005"
    run_pie_test_case("../p00075.py", input_content, expected_output)


def test_problem_p00075_1():
    input_content = "1001,50.0,1.60\n1002,60.0,1.70\n1003,70.0,1.80\n1004,80.0,1.70\n1005,90.0,1.60"
    expected_output = "1004\n1005"
    run_pie_test_case("../p00075.py", input_content, expected_output)
