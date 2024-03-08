from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00728_0():
    input_content = "3\n1000\n342\n0\n5\n2\n2\n9\n11\n932\n5\n300\n1000\n0\n200\n400\n8\n353\n242\n402\n274\n283\n132\n402\n523\n0"
    expected_output = "342\n7\n300\n326"
    run_pie_test_case("../p00728.py", input_content, expected_output)


def test_problem_p00728_1():
    input_content = "3\n1000\n342\n0\n5\n2\n2\n9\n11\n932\n5\n300\n1000\n0\n200\n400\n8\n353\n242\n402\n274\n283\n132\n402\n523\n0"
    expected_output = "342\n7\n300\n326"
    run_pie_test_case("../p00728.py", input_content, expected_output)
