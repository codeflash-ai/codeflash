from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03625_0():
    input_content = "6\n3 1 2 4 2 1"
    expected_output = "2"
    run_pie_test_case("../p03625.py", input_content, expected_output)


def test_problem_p03625_1():
    input_content = "10\n3 3 3 3 4 4 4 5 5 5"
    expected_output = "20"
    run_pie_test_case("../p03625.py", input_content, expected_output)


def test_problem_p03625_2():
    input_content = "6\n3 1 2 4 2 1"
    expected_output = "2"
    run_pie_test_case("../p03625.py", input_content, expected_output)


def test_problem_p03625_3():
    input_content = "4\n1 2 3 4"
    expected_output = "0"
    run_pie_test_case("../p03625.py", input_content, expected_output)
