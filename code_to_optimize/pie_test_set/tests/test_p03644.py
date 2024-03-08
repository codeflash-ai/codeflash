from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03644_0():
    input_content = "7"
    expected_output = "4"
    run_pie_test_case("../p03644.py", input_content, expected_output)


def test_problem_p03644_1():
    input_content = "100"
    expected_output = "64"
    run_pie_test_case("../p03644.py", input_content, expected_output)


def test_problem_p03644_2():
    input_content = "1"
    expected_output = "1"
    run_pie_test_case("../p03644.py", input_content, expected_output)


def test_problem_p03644_3():
    input_content = "7"
    expected_output = "4"
    run_pie_test_case("../p03644.py", input_content, expected_output)


def test_problem_p03644_4():
    input_content = "32"
    expected_output = "32"
    run_pie_test_case("../p03644.py", input_content, expected_output)
