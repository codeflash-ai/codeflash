from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02964_0():
    input_content = "3 2\n1 2 3"
    expected_output = "2 3"
    run_pie_test_case("../p02964.py", input_content, expected_output)


def test_problem_p02964_1():
    input_content = "11 97\n3 1 4 1 5 9 2 6 5 3 5"
    expected_output = "9 2 6"
    run_pie_test_case("../p02964.py", input_content, expected_output)


def test_problem_p02964_2():
    input_content = "5 10\n1 2 3 2 3"
    expected_output = "3"
    run_pie_test_case("../p02964.py", input_content, expected_output)


def test_problem_p02964_3():
    input_content = "6 1000000000000\n1 1 2 2 3 3"
    expected_output = ""
    run_pie_test_case("../p02964.py", input_content, expected_output)


def test_problem_p02964_4():
    input_content = "3 2\n1 2 3"
    expected_output = "2 3"
    run_pie_test_case("../p02964.py", input_content, expected_output)
