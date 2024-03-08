from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02854_0():
    input_content = "3\n2 4 3"
    expected_output = "3"
    run_pie_test_case("../p02854.py", input_content, expected_output)


def test_problem_p02854_1():
    input_content = "3\n2 4 3"
    expected_output = "3"
    run_pie_test_case("../p02854.py", input_content, expected_output)


def test_problem_p02854_2():
    input_content = "12\n100 104 102 105 103 103 101 105 104 102 104 101"
    expected_output = "0"
    run_pie_test_case("../p02854.py", input_content, expected_output)
