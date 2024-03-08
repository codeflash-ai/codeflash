from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02923_0():
    input_content = "5\n10 4 8 7 3"
    expected_output = "2"
    run_pie_test_case("../p02923.py", input_content, expected_output)


def test_problem_p02923_1():
    input_content = "5\n10 4 8 7 3"
    expected_output = "2"
    run_pie_test_case("../p02923.py", input_content, expected_output)


def test_problem_p02923_2():
    input_content = "4\n1 2 3 4"
    expected_output = "0"
    run_pie_test_case("../p02923.py", input_content, expected_output)


def test_problem_p02923_3():
    input_content = "7\n4 4 5 6 6 5 5"
    expected_output = "3"
    run_pie_test_case("../p02923.py", input_content, expected_output)
