from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02665_0():
    input_content = "3\n0 1 1 2"
    expected_output = "7"
    run_pie_test_case("../p02665.py", input_content, expected_output)


def test_problem_p02665_1():
    input_content = "4\n0 0 1 0 2"
    expected_output = "10"
    run_pie_test_case("../p02665.py", input_content, expected_output)


def test_problem_p02665_2():
    input_content = "3\n0 1 1 2"
    expected_output = "7"
    run_pie_test_case("../p02665.py", input_content, expected_output)


def test_problem_p02665_3():
    input_content = "10\n0 0 1 1 2 3 5 8 13 21 34"
    expected_output = "264"
    run_pie_test_case("../p02665.py", input_content, expected_output)


def test_problem_p02665_4():
    input_content = "2\n0 3 1"
    expected_output = "-1"
    run_pie_test_case("../p02665.py", input_content, expected_output)


def test_problem_p02665_5():
    input_content = "1\n1 1"
    expected_output = "-1"
    run_pie_test_case("../p02665.py", input_content, expected_output)
