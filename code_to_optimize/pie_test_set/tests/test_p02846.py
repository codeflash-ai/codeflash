from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02846_0():
    input_content = "1 2\n10 10\n12 4"
    expected_output = "1"
    run_pie_test_case("../p02846.py", input_content, expected_output)


def test_problem_p02846_1():
    input_content = "1 2\n10 10\n12 4"
    expected_output = "1"
    run_pie_test_case("../p02846.py", input_content, expected_output)


def test_problem_p02846_2():
    input_content = "100 1\n101 101\n102 1"
    expected_output = "infinity"
    run_pie_test_case("../p02846.py", input_content, expected_output)


def test_problem_p02846_3():
    input_content = "12000 15700\n3390000000 3810000000\n5550000000 2130000000"
    expected_output = "113"
    run_pie_test_case("../p02846.py", input_content, expected_output)
