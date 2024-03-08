from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03688_0():
    input_content = "3\n1 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_1():
    input_content = "5\n4 3 4 3 4"
    expected_output = "No"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_2():
    input_content = "3\n1 1 2"
    expected_output = "No"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_3():
    input_content = "5\n3 3 3 3 3"
    expected_output = "No"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_4():
    input_content = "3\n1 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_5():
    input_content = "4\n2 2 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03688.py", input_content, expected_output)


def test_problem_p03688_6():
    input_content = "3\n2 2 2"
    expected_output = "Yes"
    run_pie_test_case("../p03688.py", input_content, expected_output)
