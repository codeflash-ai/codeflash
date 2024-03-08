from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00090_0():
    input_content = "15\n3.14979,8.51743\n2.39506,3.84915\n2.68432,5.39095\n5.61904,9.16332\n7.85653,4.75593\n2.84021,5.41511\n1.79500,8.59211\n7.55389,8.17604\n4.70665,4.66125\n1.63470,4.42538\n7.34959,4.61981\n5.09003,8.11122\n5.24373,1.30066\n0.13517,1.83659\n7.57313,1.58150\n0"
    expected_output = "4"
    run_pie_test_case("../p00090.py", input_content, expected_output)


def test_problem_p00090_1():
    input_content = "15\n3.14979,8.51743\n2.39506,3.84915\n2.68432,5.39095\n5.61904,9.16332\n7.85653,4.75593\n2.84021,5.41511\n1.79500,8.59211\n7.55389,8.17604\n4.70665,4.66125\n1.63470,4.42538\n7.34959,4.61981\n5.09003,8.11122\n5.24373,1.30066\n0.13517,1.83659\n7.57313,1.58150\n0"
    expected_output = "4"
    run_pie_test_case("../p00090.py", input_content, expected_output)
