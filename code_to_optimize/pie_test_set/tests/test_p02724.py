from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02724_0():
    input_content = "1024"
    expected_output = "2020"
    run_pie_test_case("../p02724.py", input_content, expected_output)


def test_problem_p02724_1():
    input_content = "0"
    expected_output = "0"
    run_pie_test_case("../p02724.py", input_content, expected_output)


def test_problem_p02724_2():
    input_content = "1000000000"
    expected_output = "2000000000"
    run_pie_test_case("../p02724.py", input_content, expected_output)


def test_problem_p02724_3():
    input_content = "1024"
    expected_output = "2020"
    run_pie_test_case("../p02724.py", input_content, expected_output)
