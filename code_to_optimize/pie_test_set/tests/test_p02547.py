from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02547_0():
    input_content = "5\n1 2\n6 6\n4 4\n3 3\n3 2"
    expected_output = "Yes"
    run_pie_test_case("../p02547.py", input_content, expected_output)


def test_problem_p02547_1():
    input_content = "5\n1 2\n6 6\n4 4\n3 3\n3 2"
    expected_output = "Yes"
    run_pie_test_case("../p02547.py", input_content, expected_output)


def test_problem_p02547_2():
    input_content = "5\n1 1\n2 2\n3 4\n5 5\n6 6"
    expected_output = "No"
    run_pie_test_case("../p02547.py", input_content, expected_output)


def test_problem_p02547_3():
    input_content = "6\n1 1\n2 2\n3 3\n4 4\n5 5\n6 6"
    expected_output = "Yes"
    run_pie_test_case("../p02547.py", input_content, expected_output)
