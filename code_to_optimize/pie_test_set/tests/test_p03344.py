from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03344_0():
    input_content = "4 5\n3 1\n1 2\n4 1\n6 2\n1 2\n2 3\n2 4\n1 4\n3 4"
    expected_output = "6"
    run_pie_test_case("../p03344.py", input_content, expected_output)


def test_problem_p03344_1():
    input_content = "5 8\n6 4\n15 13\n15 19\n15 1\n20 7\n1 3\n1 4\n1 5\n2 3\n2 4\n2 5\n3 5\n4 5"
    expected_output = "44"
    run_pie_test_case("../p03344.py", input_content, expected_output)


def test_problem_p03344_2():
    input_content = "9 10\n131 2\n98 79\n242 32\n231 38\n382 82\n224 22\n140 88\n209 70\n164 64\n6 8\n1 6\n1 4\n1 3\n4 7\n4 9\n3 7\n3 9\n5 9\n2 5"
    expected_output = "582"
    run_pie_test_case("../p03344.py", input_content, expected_output)


def test_problem_p03344_3():
    input_content = "4 5\n3 1\n1 2\n4 1\n6 2\n1 2\n2 3\n2 4\n1 4\n3 4"
    expected_output = "6"
    run_pie_test_case("../p03344.py", input_content, expected_output)
