from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02952_0():
    input_content = "11"
    expected_output = "9"
    run_pie_test_case("../p02952.py", input_content, expected_output)


def test_problem_p02952_1():
    input_content = "11"
    expected_output = "9"
    run_pie_test_case("../p02952.py", input_content, expected_output)


def test_problem_p02952_2():
    input_content = "100000"
    expected_output = "90909"
    run_pie_test_case("../p02952.py", input_content, expected_output)


def test_problem_p02952_3():
    input_content = "136"
    expected_output = "46"
    run_pie_test_case("../p02952.py", input_content, expected_output)
