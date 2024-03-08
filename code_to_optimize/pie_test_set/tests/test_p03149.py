from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03149_0():
    input_content = "1 7 9 4"
    expected_output = "YES"
    run_pie_test_case("../p03149.py", input_content, expected_output)


def test_problem_p03149_1():
    input_content = "1 9 7 4"
    expected_output = "YES"
    run_pie_test_case("../p03149.py", input_content, expected_output)


def test_problem_p03149_2():
    input_content = "1 2 9 1"
    expected_output = "NO"
    run_pie_test_case("../p03149.py", input_content, expected_output)


def test_problem_p03149_3():
    input_content = "4 9 0 8"
    expected_output = "NO"
    run_pie_test_case("../p03149.py", input_content, expected_output)


def test_problem_p03149_4():
    input_content = "1 7 9 4"
    expected_output = "YES"
    run_pie_test_case("../p03149.py", input_content, expected_output)
