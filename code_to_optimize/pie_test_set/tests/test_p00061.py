from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00061_0():
    input_content = "1,20\n2,20\n3,30\n4,10\n5,10\n6,20\n0,0\n1\n2\n4\n5"
    expected_output = "2\n2\n3\n3"
    run_pie_test_case("../p00061.py", input_content, expected_output)


def test_problem_p00061_1():
    input_content = "1,20\n2,20\n3,30\n4,10\n5,10\n6,20\n0,0\n1\n2\n4\n5"
    expected_output = "2\n2\n3\n3"
    run_pie_test_case("../p00061.py", input_content, expected_output)
