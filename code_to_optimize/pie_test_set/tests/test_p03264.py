from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03264_0():
    input_content = "3"
    expected_output = "2"
    run_pie_test_case("../p03264.py", input_content, expected_output)


def test_problem_p03264_1():
    input_content = "6"
    expected_output = "9"
    run_pie_test_case("../p03264.py", input_content, expected_output)


def test_problem_p03264_2():
    input_content = "50"
    expected_output = "625"
    run_pie_test_case("../p03264.py", input_content, expected_output)


def test_problem_p03264_3():
    input_content = "11"
    expected_output = "30"
    run_pie_test_case("../p03264.py", input_content, expected_output)


def test_problem_p03264_4():
    input_content = "3"
    expected_output = "2"
    run_pie_test_case("../p03264.py", input_content, expected_output)
