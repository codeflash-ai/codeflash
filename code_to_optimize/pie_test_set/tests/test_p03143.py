from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03143_0():
    input_content = "4 4\n2 3 5 7\n1 2 7\n1 3 9\n2 3 12\n3 4 18"
    expected_output = "2"
    run_pie_test_case("../p03143.py", input_content, expected_output)


def test_problem_p03143_1():
    input_content = "10 9\n81 16 73 7 2 61 86 38 90 28\n6 8 725\n3 10 12\n1 4 558\n4 9 615\n5 6 942\n8 9 918\n2 7 720\n4 7 292\n7 10 414"
    expected_output = "8"
    run_pie_test_case("../p03143.py", input_content, expected_output)


def test_problem_p03143_2():
    input_content = "4 4\n2 3 5 7\n1 2 7\n1 3 9\n2 3 12\n3 4 18"
    expected_output = "2"
    run_pie_test_case("../p03143.py", input_content, expected_output)


def test_problem_p03143_3():
    input_content = "6 10\n4 4 1 1 1 7\n3 5 19\n2 5 20\n4 5 8\n1 6 16\n2 3 9\n3 6 16\n3 4 1\n2 6 20\n2 4 19\n1 2 9"
    expected_output = "4"
    run_pie_test_case("../p03143.py", input_content, expected_output)
