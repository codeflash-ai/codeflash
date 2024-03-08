from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02546_0():
    input_content = "apple"
    expected_output = "apples"
    run_pie_test_case("../p02546.py", input_content, expected_output)


def test_problem_p02546_1():
    input_content = "box"
    expected_output = "boxs"
    run_pie_test_case("../p02546.py", input_content, expected_output)


def test_problem_p02546_2():
    input_content = "bus"
    expected_output = "buses"
    run_pie_test_case("../p02546.py", input_content, expected_output)


def test_problem_p02546_3():
    input_content = "apple"
    expected_output = "apples"
    run_pie_test_case("../p02546.py", input_content, expected_output)
