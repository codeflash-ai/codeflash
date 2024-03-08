from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02798_0():
    input_content = "3\n3 4 3\n3 2 3"
    expected_output = "1"
    run_pie_test_case("../p02798.py", input_content, expected_output)


def test_problem_p02798_1():
    input_content = "5\n28 15 22 43 31\n20 22 43 33 32"
    expected_output = "-1"
    run_pie_test_case("../p02798.py", input_content, expected_output)


def test_problem_p02798_2():
    input_content = "4\n1 2 3 4\n5 6 7 8"
    expected_output = "0"
    run_pie_test_case("../p02798.py", input_content, expected_output)


def test_problem_p02798_3():
    input_content = "3\n3 4 3\n3 2 3"
    expected_output = "1"
    run_pie_test_case("../p02798.py", input_content, expected_output)


def test_problem_p02798_4():
    input_content = "5\n4 46 6 38 43\n33 15 18 27 37"
    expected_output = "3"
    run_pie_test_case("../p02798.py", input_content, expected_output)


def test_problem_p02798_5():
    input_content = "2\n2 1\n1 2"
    expected_output = "-1"
    run_pie_test_case("../p02798.py", input_content, expected_output)
