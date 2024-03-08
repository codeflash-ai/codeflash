from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03407_0():
    input_content = "50 100 120"
    expected_output = "Yes"
    run_pie_test_case("../p03407.py", input_content, expected_output)


def test_problem_p03407_1():
    input_content = "500 100 1000"
    expected_output = "No"
    run_pie_test_case("../p03407.py", input_content, expected_output)


def test_problem_p03407_2():
    input_content = "19 123 143"
    expected_output = "No"
    run_pie_test_case("../p03407.py", input_content, expected_output)


def test_problem_p03407_3():
    input_content = "19 123 142"
    expected_output = "Yes"
    run_pie_test_case("../p03407.py", input_content, expected_output)


def test_problem_p03407_4():
    input_content = "50 100 120"
    expected_output = "Yes"
    run_pie_test_case("../p03407.py", input_content, expected_output)
