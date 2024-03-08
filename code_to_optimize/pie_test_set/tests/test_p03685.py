from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03685_0():
    input_content = "4 2 3\n0 1 3 1\n1 1 4 1\n2 0 2 2"
    expected_output = "YES"
    run_pie_test_case("../p03685.py", input_content, expected_output)


def test_problem_p03685_1():
    input_content = "4 2 3\n0 1 3 1\n1 1 4 1\n2 0 2 2"
    expected_output = "YES"
    run_pie_test_case("../p03685.py", input_content, expected_output)


def test_problem_p03685_2():
    input_content = "5 5 7\n0 0 2 4\n2 3 4 5\n3 5 5 2\n5 5 5 4\n0 3 5 1\n2 2 4 4\n0 5 4 1"
    expected_output = "YES"
    run_pie_test_case("../p03685.py", input_content, expected_output)


def test_problem_p03685_3():
    input_content = "2 2 4\n0 0 2 2\n2 0 0 1\n0 2 1 2\n1 1 2 1"
    expected_output = "NO"
    run_pie_test_case("../p03685.py", input_content, expected_output)


def test_problem_p03685_4():
    input_content = "1 1 2\n0 0 1 1\n1 0 0 1"
    expected_output = "NO"
    run_pie_test_case("../p03685.py", input_content, expected_output)
