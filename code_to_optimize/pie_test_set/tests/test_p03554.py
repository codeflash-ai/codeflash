from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03554_0():
    input_content = "3\n1 0 1\n1\n1 3"
    expected_output = "1"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_1():
    input_content = "5\n0 1 0 1 0\n1\n1 5"
    expected_output = "2"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_2():
    input_content = "3\n1 0 1\n2\n1 1\n3 3"
    expected_output = "0"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_3():
    input_content = "10\n0 0 0 1 0 0 1 1 1 0\n7\n1 4\n2 5\n1 3\n6 7\n9 9\n1 5\n7 9"
    expected_output = "1"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_4():
    input_content = "3\n1 0 1\n1\n1 3"
    expected_output = "1"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_5():
    input_content = "3\n1 0 1\n2\n1 1\n2 3"
    expected_output = "1"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_6():
    input_content = (
        "15\n1 1 0 0 0 0 0 0 1 0 1 1 1 0 0\n9\n4 10\n13 14\n1 7\n4 14\n9 11\n2 6\n7 8\n3 12\n7 13"
    )
    expected_output = "5"
    run_pie_test_case("../p03554.py", input_content, expected_output)


def test_problem_p03554_7():
    input_content = "9\n0 1 0 1 1 1 0 1 0\n3\n1 4\n5 8\n6 7"
    expected_output = "3"
    run_pie_test_case("../p03554.py", input_content, expected_output)
