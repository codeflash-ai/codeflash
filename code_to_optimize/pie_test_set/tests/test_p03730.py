from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03730_0():
    input_content = "7 5 1"
    expected_output = "YES"
    run_pie_test_case("../p03730.py", input_content, expected_output)


def test_problem_p03730_1():
    input_content = "7 5 1"
    expected_output = "YES"
    run_pie_test_case("../p03730.py", input_content, expected_output)


def test_problem_p03730_2():
    input_content = "2 2 1"
    expected_output = "NO"
    run_pie_test_case("../p03730.py", input_content, expected_output)


def test_problem_p03730_3():
    input_content = "77 42 36"
    expected_output = "NO"
    run_pie_test_case("../p03730.py", input_content, expected_output)


def test_problem_p03730_4():
    input_content = "40 98 58"
    expected_output = "YES"
    run_pie_test_case("../p03730.py", input_content, expected_output)


def test_problem_p03730_5():
    input_content = "1 100 97"
    expected_output = "YES"
    run_pie_test_case("../p03730.py", input_content, expected_output)
