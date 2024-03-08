from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03699_0():
    input_content = "3\n5\n10\n15"
    expected_output = "25"
    run_pie_test_case("../p03699.py", input_content, expected_output)


def test_problem_p03699_1():
    input_content = "3\n10\n10\n15"
    expected_output = "35"
    run_pie_test_case("../p03699.py", input_content, expected_output)


def test_problem_p03699_2():
    input_content = "3\n10\n20\n30"
    expected_output = "0"
    run_pie_test_case("../p03699.py", input_content, expected_output)


def test_problem_p03699_3():
    input_content = "3\n5\n10\n15"
    expected_output = "25"
    run_pie_test_case("../p03699.py", input_content, expected_output)
