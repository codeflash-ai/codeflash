from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00211_0():
    input_content = "2\n4 3\n5 4\n5\n789 289\n166 46\n9 4\n617 252\n972 303\n2\n8 5\n32 20\n0"
    expected_output = "15\n16\n1598397732\n1209243492\n1939462992\n1782294192\n1360317793\n1\n1"
    run_pie_test_case("../p00211.py", input_content, expected_output)


def test_problem_p00211_1():
    input_content = "2\n4 3\n5 4\n5\n789 289\n166 46\n9 4\n617 252\n972 303\n2\n8 5\n32 20\n0"
    expected_output = "15\n16\n1598397732\n1209243492\n1939462992\n1782294192\n1360317793\n1\n1"
    run_pie_test_case("../p00211.py", input_content, expected_output)
