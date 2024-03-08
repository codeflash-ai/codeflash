from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02617_0():
    input_content = "3\n1 3\n2 3"
    expected_output = "7"
    run_pie_test_case("../p02617.py", input_content, expected_output)


def test_problem_p02617_1():
    input_content = "2\n1 2"
    expected_output = "3"
    run_pie_test_case("../p02617.py", input_content, expected_output)


def test_problem_p02617_2():
    input_content = "10\n5 3\n5 7\n8 9\n1 9\n9 10\n8 4\n7 4\n6 10\n7 2"
    expected_output = "113"
    run_pie_test_case("../p02617.py", input_content, expected_output)


def test_problem_p02617_3():
    input_content = "3\n1 3\n2 3"
    expected_output = "7"
    run_pie_test_case("../p02617.py", input_content, expected_output)
