from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02603_0():
    input_content = "7\n100 130 130 130 115 115 150"
    expected_output = "1685"
    run_pie_test_case("../p02603.py", input_content, expected_output)


def test_problem_p02603_1():
    input_content = "2\n157 193"
    expected_output = "1216"
    run_pie_test_case("../p02603.py", input_content, expected_output)


def test_problem_p02603_2():
    input_content = "7\n100 130 130 130 115 115 150"
    expected_output = "1685"
    run_pie_test_case("../p02603.py", input_content, expected_output)


def test_problem_p02603_3():
    input_content = "6\n200 180 160 140 120 100"
    expected_output = "1000"
    run_pie_test_case("../p02603.py", input_content, expected_output)
