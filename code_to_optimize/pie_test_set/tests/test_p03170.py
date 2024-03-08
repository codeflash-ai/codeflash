from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03170_0():
    input_content = "2 4\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_1():
    input_content = "1 100000\n1"
    expected_output = "Second"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_2():
    input_content = "3 21\n1 2 3"
    expected_output = "First"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_3():
    input_content = "2 5\n2 3"
    expected_output = "Second"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_4():
    input_content = "2 4\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_5():
    input_content = "2 7\n2 3"
    expected_output = "First"
    run_pie_test_case("../p03170.py", input_content, expected_output)


def test_problem_p03170_6():
    input_content = "3 20\n1 2 3"
    expected_output = "Second"
    run_pie_test_case("../p03170.py", input_content, expected_output)
