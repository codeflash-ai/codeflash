from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00068_0():
    input_content = "4\n1.0,0.0\n0.0,1.0\n2.0,1.0\n1.0,2.0\n9\n-509.94,892.63\n567.62,639.99\n-859.32,-64.84\n-445.99,383.69\n667.54,430.49\n551.12,828.21\n-940.2,-877.2\n-361.62,-970\n-125.42,-178.48\n0"
    expected_output = "0\n3"
    run_pie_test_case("../p00068.py", input_content, expected_output)


def test_problem_p00068_1():
    input_content = "4\n1.0,0.0\n0.0,1.0\n2.0,1.0\n1.0,2.0\n9\n-509.94,892.63\n567.62,639.99\n-859.32,-64.84\n-445.99,383.69\n667.54,430.49\n551.12,828.21\n-940.2,-877.2\n-361.62,-970\n-125.42,-178.48\n0"
    expected_output = "0\n3"
    run_pie_test_case("../p00068.py", input_content, expected_output)
