from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00760_0():
    input_content = "8\n1 1 1\n344 3 1\n696 5 1\n182 9 5\n998 8 7\n344 2 19\n696 4 19\n999 10 20"
    expected_output = "196470\n128976\n59710\n160715\n252\n128977\n59712\n1"
    run_pie_test_case("../p00760.py", input_content, expected_output)


def test_problem_p00760_1():
    input_content = "8\n1 1 1\n344 3 1\n696 5 1\n182 9 5\n998 8 7\n344 2 19\n696 4 19\n999 10 20"
    expected_output = "196470\n128976\n59710\n160715\n252\n128977\n59712\n1"
    run_pie_test_case("../p00760.py", input_content, expected_output)
