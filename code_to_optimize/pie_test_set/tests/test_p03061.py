from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03061_0():
    input_content = "3\n7 6 8"
    expected_output = "2"
    run_pie_test_case("../p03061.py", input_content, expected_output)


def test_problem_p03061_1():
    input_content = "3\n7 6 8"
    expected_output = "2"
    run_pie_test_case("../p03061.py", input_content, expected_output)


def test_problem_p03061_2():
    input_content = "2\n1000000000 1000000000"
    expected_output = "1000000000"
    run_pie_test_case("../p03061.py", input_content, expected_output)


def test_problem_p03061_3():
    input_content = "3\n12 15 18"
    expected_output = "6"
    run_pie_test_case("../p03061.py", input_content, expected_output)
