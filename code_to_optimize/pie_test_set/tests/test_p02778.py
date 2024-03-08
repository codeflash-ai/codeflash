from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02778_0():
    input_content = "sardine"
    expected_output = "xxxxxxx"
    run_pie_test_case("../p02778.py", input_content, expected_output)


def test_problem_p02778_1():
    input_content = "sardine"
    expected_output = "xxxxxxx"
    run_pie_test_case("../p02778.py", input_content, expected_output)


def test_problem_p02778_2():
    input_content = "xxxx"
    expected_output = "xxxx"
    run_pie_test_case("../p02778.py", input_content, expected_output)


def test_problem_p02778_3():
    input_content = "gone"
    expected_output = "xxxx"
    run_pie_test_case("../p02778.py", input_content, expected_output)
