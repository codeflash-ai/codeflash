from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02841_0():
    input_content = "11 16\n11 17"
    expected_output = "0"
    run_pie_test_case("../p02841.py", input_content, expected_output)


def test_problem_p02841_1():
    input_content = "11 16\n11 17"
    expected_output = "0"
    run_pie_test_case("../p02841.py", input_content, expected_output)


def test_problem_p02841_2():
    input_content = "11 30\n12 1"
    expected_output = "1"
    run_pie_test_case("../p02841.py", input_content, expected_output)
