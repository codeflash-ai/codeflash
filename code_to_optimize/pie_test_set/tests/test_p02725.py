from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02725_0():
    input_content = "20 3\n5 10 15"
    expected_output = "10"
    run_pie_test_case("../p02725.py", input_content, expected_output)


def test_problem_p02725_1():
    input_content = "20 3\n5 10 15"
    expected_output = "10"
    run_pie_test_case("../p02725.py", input_content, expected_output)


def test_problem_p02725_2():
    input_content = "20 3\n0 5 15"
    expected_output = "10"
    run_pie_test_case("../p02725.py", input_content, expected_output)
