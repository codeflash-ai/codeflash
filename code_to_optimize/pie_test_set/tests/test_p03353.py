from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03353_0():
    input_content = "aba\n4"
    expected_output = "b"
    run_pie_test_case("../p03353.py", input_content, expected_output)


def test_problem_p03353_1():
    input_content = "aba\n4"
    expected_output = "b"
    run_pie_test_case("../p03353.py", input_content, expected_output)


def test_problem_p03353_2():
    input_content = "atcoderandatcodeer\n5"
    expected_output = "andat"
    run_pie_test_case("../p03353.py", input_content, expected_output)


def test_problem_p03353_3():
    input_content = "z\n1"
    expected_output = "z"
    run_pie_test_case("../p03353.py", input_content, expected_output)
