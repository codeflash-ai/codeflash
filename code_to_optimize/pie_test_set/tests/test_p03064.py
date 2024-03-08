from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03064_0():
    input_content = "4\n1\n1\n1\n2"
    expected_output = "18"
    run_pie_test_case("../p03064.py", input_content, expected_output)


def test_problem_p03064_1():
    input_content = "4\n1\n1\n1\n2"
    expected_output = "18"
    run_pie_test_case("../p03064.py", input_content, expected_output)


def test_problem_p03064_2():
    input_content = "6\n1\n3\n2\n3\n5\n2"
    expected_output = "150"
    run_pie_test_case("../p03064.py", input_content, expected_output)


def test_problem_p03064_3():
    input_content = "20\n3\n1\n4\n1\n5\n9\n2\n6\n5\n3\n5\n8\n9\n7\n9\n3\n2\n3\n8\n4"
    expected_output = "563038556"
    run_pie_test_case("../p03064.py", input_content, expected_output)
