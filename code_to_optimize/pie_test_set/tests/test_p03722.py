from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03722_0():
    input_content = "3 3\n1 2 4\n2 3 3\n1 3 5"
    expected_output = "7"
    run_pie_test_case("../p03722.py", input_content, expected_output)


def test_problem_p03722_1():
    input_content = "2 2\n1 2 1\n2 1 1"
    expected_output = "inf"
    run_pie_test_case("../p03722.py", input_content, expected_output)


def test_problem_p03722_2():
    input_content = "3 3\n1 2 4\n2 3 3\n1 3 5"
    expected_output = "7"
    run_pie_test_case("../p03722.py", input_content, expected_output)


def test_problem_p03722_3():
    input_content = (
        "6 5\n1 2 -1000000000\n2 3 -1000000000\n3 4 -1000000000\n4 5 -1000000000\n5 6 -1000000000"
    )
    expected_output = "-5000000000"
    run_pie_test_case("../p03722.py", input_content, expected_output)
