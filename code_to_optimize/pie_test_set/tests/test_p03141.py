from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03141_0():
    input_content = "3\n10 10\n20 20\n30 30"
    expected_output = "20"
    run_pie_test_case("../p03141.py", input_content, expected_output)


def test_problem_p03141_1():
    input_content = "3\n20 10\n20 20\n20 30"
    expected_output = "20"
    run_pie_test_case("../p03141.py", input_content, expected_output)


def test_problem_p03141_2():
    input_content = "3\n10 10\n20 20\n30 30"
    expected_output = "20"
    run_pie_test_case("../p03141.py", input_content, expected_output)


def test_problem_p03141_3():
    input_content = (
        "6\n1 1000000000\n1 1000000000\n1 1000000000\n1 1000000000\n1 1000000000\n1 1000000000"
    )
    expected_output = "-2999999997"
    run_pie_test_case("../p03141.py", input_content, expected_output)
