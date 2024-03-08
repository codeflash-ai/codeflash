from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02709_0():
    input_content = "4\n1 3 4 2"
    expected_output = "20"
    run_pie_test_case("../p02709.py", input_content, expected_output)


def test_problem_p02709_1():
    input_content = "4\n1 3 4 2"
    expected_output = "20"
    run_pie_test_case("../p02709.py", input_content, expected_output)


def test_problem_p02709_2():
    input_content = "6\n8 6 9 1 2 1"
    expected_output = "85"
    run_pie_test_case("../p02709.py", input_content, expected_output)


def test_problem_p02709_3():
    input_content = "6\n5 5 6 1 1 1"
    expected_output = "58"
    run_pie_test_case("../p02709.py", input_content, expected_output)
