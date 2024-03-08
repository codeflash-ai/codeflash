from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00049_0():
    input_content = "1,B\n2,A\n3,B\n4,AB\n5,B\n6,O\n7,A\n8,O\n9,AB\n10,A\n11,A\n12,B\n13,AB\n14,A"
    expected_output = "5\n4\n3\n2"
    run_pie_test_case("../p00049.py", input_content, expected_output)


def test_problem_p00049_1():
    input_content = "1,B\n2,A\n3,B\n4,AB\n5,B\n6,O\n7,A\n8,O\n9,AB\n10,A\n11,A\n12,B\n13,AB\n14,A"
    expected_output = "5\n4\n3\n2"
    run_pie_test_case("../p00049.py", input_content, expected_output)
