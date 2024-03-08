from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00137_0():
    input_content = "2\n123\n567"
    expected_output = "Case 1:\n151\n228\n519\n2693\n2522\n3604\n9888\n7725\n6756\n6435\nCase 2:\n3214\n3297\n8702\n7248\n5335\n4622\n3628\n1623\n6341\n2082"
    run_pie_test_case("../p00137.py", input_content, expected_output)


def test_problem_p00137_1():
    input_content = "2\n123\n567"
    expected_output = "Case 1:\n151\n228\n519\n2693\n2522\n3604\n9888\n7725\n6756\n6435\nCase 2:\n3214\n3297\n8702\n7248\n5335\n4622\n3628\n1623\n6341\n2082"
    run_pie_test_case("../p00137.py", input_content, expected_output)
