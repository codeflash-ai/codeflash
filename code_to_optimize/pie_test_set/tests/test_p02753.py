from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02753_0():
    input_content = "ABA"
    expected_output = "Yes"
    run_pie_test_case("../p02753.py", input_content, expected_output)


def test_problem_p02753_1():
    input_content = "ABA"
    expected_output = "Yes"
    run_pie_test_case("../p02753.py", input_content, expected_output)


def test_problem_p02753_2():
    input_content = "BBA"
    expected_output = "Yes"
    run_pie_test_case("../p02753.py", input_content, expected_output)


def test_problem_p02753_3():
    input_content = "BBB"
    expected_output = "No"
    run_pie_test_case("../p02753.py", input_content, expected_output)
