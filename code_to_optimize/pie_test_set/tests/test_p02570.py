from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02570_0():
    input_content = "1000 15 80"
    expected_output = "Yes"
    run_pie_test_case("../p02570.py", input_content, expected_output)


def test_problem_p02570_1():
    input_content = "2000 20 100"
    expected_output = "Yes"
    run_pie_test_case("../p02570.py", input_content, expected_output)


def test_problem_p02570_2():
    input_content = "10000 1 1"
    expected_output = "No"
    run_pie_test_case("../p02570.py", input_content, expected_output)


def test_problem_p02570_3():
    input_content = "1000 15 80"
    expected_output = "Yes"
    run_pie_test_case("../p02570.py", input_content, expected_output)
