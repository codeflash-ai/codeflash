from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03568_0():
    input_content = "2\n2 3"
    expected_output = "7"
    run_pie_test_case("../p03568.py", input_content, expected_output)


def test_problem_p03568_1():
    input_content = "10\n90 52 56 71 44 8 13 30 57 84"
    expected_output = "58921"
    run_pie_test_case("../p03568.py", input_content, expected_output)


def test_problem_p03568_2():
    input_content = "2\n2 3"
    expected_output = "7"
    run_pie_test_case("../p03568.py", input_content, expected_output)


def test_problem_p03568_3():
    input_content = "1\n100"
    expected_output = "1"
    run_pie_test_case("../p03568.py", input_content, expected_output)


def test_problem_p03568_4():
    input_content = "3\n3 3 3"
    expected_output = "26"
    run_pie_test_case("../p03568.py", input_content, expected_output)
