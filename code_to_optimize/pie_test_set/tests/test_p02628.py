from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02628_0():
    input_content = "5 3\n50 100 80 120 80"
    expected_output = "210"
    run_pie_test_case("../p02628.py", input_content, expected_output)


def test_problem_p02628_1():
    input_content = "1 1\n1000"
    expected_output = "1000"
    run_pie_test_case("../p02628.py", input_content, expected_output)


def test_problem_p02628_2():
    input_content = "5 3\n50 100 80 120 80"
    expected_output = "210"
    run_pie_test_case("../p02628.py", input_content, expected_output)
