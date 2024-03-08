from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03452_0():
    input_content = "3 3\n1 2 1\n2 3 1\n1 3 2"
    expected_output = "Yes"
    run_pie_test_case("../p03452.py", input_content, expected_output)


def test_problem_p03452_1():
    input_content = "3 3\n1 2 1\n2 3 1\n1 3 5"
    expected_output = "No"
    run_pie_test_case("../p03452.py", input_content, expected_output)


def test_problem_p03452_2():
    input_content = "100 0"
    expected_output = "Yes"
    run_pie_test_case("../p03452.py", input_content, expected_output)


def test_problem_p03452_3():
    input_content = "10 3\n8 7 100\n7 9 100\n9 8 100"
    expected_output = "No"
    run_pie_test_case("../p03452.py", input_content, expected_output)


def test_problem_p03452_4():
    input_content = "3 3\n1 2 1\n2 3 1\n1 3 2"
    expected_output = "Yes"
    run_pie_test_case("../p03452.py", input_content, expected_output)


def test_problem_p03452_5():
    input_content = "4 3\n2 1 1\n2 3 5\n3 4 2"
    expected_output = "Yes"
    run_pie_test_case("../p03452.py", input_content, expected_output)
