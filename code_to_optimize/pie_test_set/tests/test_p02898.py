from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02898_0():
    input_content = "4 150\n150 140 100 200"
    expected_output = "2"
    run_pie_test_case("../p02898.py", input_content, expected_output)


def test_problem_p02898_1():
    input_content = "5 1\n100 200 300 400 500"
    expected_output = "5"
    run_pie_test_case("../p02898.py", input_content, expected_output)


def test_problem_p02898_2():
    input_content = "4 150\n150 140 100 200"
    expected_output = "2"
    run_pie_test_case("../p02898.py", input_content, expected_output)


def test_problem_p02898_3():
    input_content = "1 500\n499"
    expected_output = "0"
    run_pie_test_case("../p02898.py", input_content, expected_output)
