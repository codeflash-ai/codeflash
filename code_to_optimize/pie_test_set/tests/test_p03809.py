from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03809_0():
    input_content = "5\n1 2 1 1 2\n2 4\n5 2\n3 2\n1 3"
    expected_output = "YES"
    run_pie_test_case("../p03809.py", input_content, expected_output)


def test_problem_p03809_1():
    input_content = "6\n3 2 2 2 2 2\n1 2\n2 3\n1 4\n1 5\n4 6"
    expected_output = "YES"
    run_pie_test_case("../p03809.py", input_content, expected_output)


def test_problem_p03809_2():
    input_content = "5\n1 2 1 1 2\n2 4\n5 2\n3 2\n1 3"
    expected_output = "YES"
    run_pie_test_case("../p03809.py", input_content, expected_output)


def test_problem_p03809_3():
    input_content = "3\n1 2 1\n1 2\n2 3"
    expected_output = "NO"
    run_pie_test_case("../p03809.py", input_content, expected_output)
