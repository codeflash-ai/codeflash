from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03386_0():
    input_content = "3 8 2"
    expected_output = "3\n4\n7\n8"
    run_pie_test_case("../p03386.py", input_content, expected_output)


def test_problem_p03386_1():
    input_content = "2 9 100"
    expected_output = "2\n3\n4\n5\n6\n7\n8\n9"
    run_pie_test_case("../p03386.py", input_content, expected_output)


def test_problem_p03386_2():
    input_content = "3 8 2"
    expected_output = "3\n4\n7\n8"
    run_pie_test_case("../p03386.py", input_content, expected_output)


def test_problem_p03386_3():
    input_content = "4 8 3"
    expected_output = "4\n5\n6\n7\n8"
    run_pie_test_case("../p03386.py", input_content, expected_output)
