from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03233_0():
    input_content = "3\n1 5\n4 2\n6 3"
    expected_output = "7"
    run_pie_test_case("../p03233.py", input_content, expected_output)


def test_problem_p03233_1():
    input_content = "3\n1 5\n4 2\n6 3"
    expected_output = "7"
    run_pie_test_case("../p03233.py", input_content, expected_output)


def test_problem_p03233_2():
    input_content = "6\n19 92\n64 64\n78 48\n57 33\n73 6\n95 73"
    expected_output = "227"
    run_pie_test_case("../p03233.py", input_content, expected_output)


def test_problem_p03233_3():
    input_content = "4\n1 5\n2 6\n3 7\n4 8"
    expected_output = "10"
    run_pie_test_case("../p03233.py", input_content, expected_output)
