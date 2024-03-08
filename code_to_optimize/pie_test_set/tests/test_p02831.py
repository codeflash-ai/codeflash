from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02831_0():
    input_content = "2 3"
    expected_output = "6"
    run_pie_test_case("../p02831.py", input_content, expected_output)


def test_problem_p02831_1():
    input_content = "100000 99999"
    expected_output = "9999900000"
    run_pie_test_case("../p02831.py", input_content, expected_output)


def test_problem_p02831_2():
    input_content = "123 456"
    expected_output = "18696"
    run_pie_test_case("../p02831.py", input_content, expected_output)


def test_problem_p02831_3():
    input_content = "2 3"
    expected_output = "6"
    run_pie_test_case("../p02831.py", input_content, expected_output)
