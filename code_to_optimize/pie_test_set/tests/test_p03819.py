from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03819_0():
    input_content = "3 3\n1 2\n2 3\n3 3"
    expected_output = "3\n2\n2"
    run_pie_test_case("../p03819.py", input_content, expected_output)


def test_problem_p03819_1():
    input_content = "3 3\n1 2\n2 3\n3 3"
    expected_output = "3\n2\n2"
    run_pie_test_case("../p03819.py", input_content, expected_output)


def test_problem_p03819_2():
    input_content = "7 9\n1 7\n5 9\n5 7\n5 9\n1 1\n6 8\n3 4"
    expected_output = "7\n6\n6\n5\n4\n5\n5\n3\n2"
    run_pie_test_case("../p03819.py", input_content, expected_output)
