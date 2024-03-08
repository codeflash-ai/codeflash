from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03988_0():
    input_content = "5\n3 2 2 3 3"
    expected_output = "Possible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_1():
    input_content = "6\n1 1 1 1 1 5"
    expected_output = "Impossible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_2():
    input_content = "3\n1 1 2"
    expected_output = "Impossible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_3():
    input_content = "5\n3 2 2 3 3"
    expected_output = "Possible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_4():
    input_content = "5\n4 3 2 3 4"
    expected_output = "Possible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_5():
    input_content = "10\n1 2 2 2 2 2 2 2 2 2"
    expected_output = "Possible"
    run_pie_test_case("../p03988.py", input_content, expected_output)


def test_problem_p03988_6():
    input_content = "10\n1 1 2 2 2 2 2 2 2 2"
    expected_output = "Impossible"
    run_pie_test_case("../p03988.py", input_content, expected_output)
