from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02685_0():
    input_content = "3 2 1"
    expected_output = "6"
    run_pie_test_case("../p02685.py", input_content, expected_output)


def test_problem_p02685_1():
    input_content = "100 100 0"
    expected_output = "73074801"
    run_pie_test_case("../p02685.py", input_content, expected_output)


def test_problem_p02685_2():
    input_content = "3 2 1"
    expected_output = "6"
    run_pie_test_case("../p02685.py", input_content, expected_output)


def test_problem_p02685_3():
    input_content = "60522 114575 7559"
    expected_output = "479519525"
    run_pie_test_case("../p02685.py", input_content, expected_output)
