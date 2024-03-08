from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02710_0():
    input_content = "3\n1 2 1\n1 2\n2 3"
    expected_output = "5\n4\n0"
    run_pie_test_case("../p02710.py", input_content, expected_output)


def test_problem_p02710_1():
    input_content = "3\n1 2 1\n1 2\n2 3"
    expected_output = "5\n4\n0"
    run_pie_test_case("../p02710.py", input_content, expected_output)


def test_problem_p02710_2():
    input_content = "1\n1"
    expected_output = "1"
    run_pie_test_case("../p02710.py", input_content, expected_output)


def test_problem_p02710_3():
    input_content = "2\n1 2\n1 2"
    expected_output = "2\n2"
    run_pie_test_case("../p02710.py", input_content, expected_output)


def test_problem_p02710_4():
    input_content = "8\n2 7 2 5 4 1 7 5\n3 1\n1 2\n2 7\n4 5\n5 6\n6 8\n7 8"
    expected_output = "18\n15\n0\n14\n23\n0\n23\n0"
    run_pie_test_case("../p02710.py", input_content, expected_output)


def test_problem_p02710_5():
    input_content = "5\n1 2 3 4 5\n1 2\n2 3\n3 4\n3 5"
    expected_output = "5\n8\n10\n5\n5"
    run_pie_test_case("../p02710.py", input_content, expected_output)
