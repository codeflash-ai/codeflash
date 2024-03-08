from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02775_0():
    input_content = "36"
    expected_output = "8"
    run_pie_test_case("../p02775.py", input_content, expected_output)


def test_problem_p02775_1():
    input_content = "91"
    expected_output = "3"
    run_pie_test_case("../p02775.py", input_content, expected_output)


def test_problem_p02775_2():
    input_content = "314159265358979323846264338327950288419716939937551058209749445923078164062862089986280348253421170"
    expected_output = "243"
    run_pie_test_case("../p02775.py", input_content, expected_output)


def test_problem_p02775_3():
    input_content = "36"
    expected_output = "8"
    run_pie_test_case("../p02775.py", input_content, expected_output)
