from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02777_0():
    input_content = "red blue\n3 4\nred"
    expected_output = "2 4"
    run_pie_test_case("../p02777.py", input_content, expected_output)


def test_problem_p02777_1():
    input_content = "red blue\n3 4\nred"
    expected_output = "2 4"
    run_pie_test_case("../p02777.py", input_content, expected_output)


def test_problem_p02777_2():
    input_content = "red blue\n5 5\nblue"
    expected_output = "5 4"
    run_pie_test_case("../p02777.py", input_content, expected_output)
