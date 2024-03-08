from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02582_0():
    input_content = "RRS"
    expected_output = "2"
    run_pie_test_case("../p02582.py", input_content, expected_output)


def test_problem_p02582_1():
    input_content = "RSR"
    expected_output = "1"
    run_pie_test_case("../p02582.py", input_content, expected_output)


def test_problem_p02582_2():
    input_content = "RRS"
    expected_output = "2"
    run_pie_test_case("../p02582.py", input_content, expected_output)


def test_problem_p02582_3():
    input_content = "SSS"
    expected_output = "0"
    run_pie_test_case("../p02582.py", input_content, expected_output)
