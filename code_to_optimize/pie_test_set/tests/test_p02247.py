from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02247_0():
    input_content = "aabaaa\naa"
    expected_output = "0\n3\n4"
    run_pie_test_case("../p02247.py", input_content, expected_output)


def test_problem_p02247_1():
    input_content = "aabaaa\naa"
    expected_output = "0\n3\n4"
    run_pie_test_case("../p02247.py", input_content, expected_output)


def test_problem_p02247_2():
    input_content = "xyzz\nyz"
    expected_output = "1"
    run_pie_test_case("../p02247.py", input_content, expected_output)
