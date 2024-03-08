from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03357_0():
    input_content = "3\nB 1\nW 2\nB 3\nW 1\nW 3\nB 2"
    expected_output = "4"
    run_pie_test_case("../p03357.py", input_content, expected_output)


def test_problem_p03357_1():
    input_content = "9\nW 3\nB 1\nB 4\nW 1\nB 5\nW 9\nW 2\nB 6\nW 5\nB 3\nW 8\nB 9\nW 7\nB 2\nB 8\nW 4\nW 6\nB 7"
    expected_output = "41"
    run_pie_test_case("../p03357.py", input_content, expected_output)


def test_problem_p03357_2():
    input_content = "3\nB 1\nW 2\nB 3\nW 1\nW 3\nB 2"
    expected_output = "4"
    run_pie_test_case("../p03357.py", input_content, expected_output)


def test_problem_p03357_3():
    input_content = "4\nB 4\nW 4\nB 3\nW 3\nB 2\nW 2\nB 1\nW 1"
    expected_output = "18"
    run_pie_test_case("../p03357.py", input_content, expected_output)
