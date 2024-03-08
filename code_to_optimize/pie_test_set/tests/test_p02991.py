from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02991_0():
    input_content = "4 4\n1 2\n2 3\n3 4\n4 1\n1 3"
    expected_output = "2"
    run_pie_test_case("../p02991.py", input_content, expected_output)


def test_problem_p02991_1():
    input_content = "4 4\n1 2\n2 3\n3 4\n4 1\n1 3"
    expected_output = "2"
    run_pie_test_case("../p02991.py", input_content, expected_output)


def test_problem_p02991_2():
    input_content = "6 8\n1 2\n2 3\n3 4\n4 5\n5 1\n1 4\n1 5\n4 6\n1 6"
    expected_output = "2"
    run_pie_test_case("../p02991.py", input_content, expected_output)


def test_problem_p02991_3():
    input_content = "3 3\n1 2\n2 3\n3 1\n1 2"
    expected_output = "-1"
    run_pie_test_case("../p02991.py", input_content, expected_output)


def test_problem_p02991_4():
    input_content = "2 0\n1 2"
    expected_output = "-1"
    run_pie_test_case("../p02991.py", input_content, expected_output)
