from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02951_0():
    input_content = "6 4 3"
    expected_output = "1"
    run_pie_test_case("../p02951.py", input_content, expected_output)


def test_problem_p02951_1():
    input_content = "12 3 7"
    expected_output = "0"
    run_pie_test_case("../p02951.py", input_content, expected_output)


def test_problem_p02951_2():
    input_content = "8 3 9"
    expected_output = "4"
    run_pie_test_case("../p02951.py", input_content, expected_output)


def test_problem_p02951_3():
    input_content = "6 4 3"
    expected_output = "1"
    run_pie_test_case("../p02951.py", input_content, expected_output)
