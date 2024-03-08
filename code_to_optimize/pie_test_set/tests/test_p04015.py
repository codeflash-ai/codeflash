from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04015_0():
    input_content = "4 8\n7 9 8 9"
    expected_output = "5"
    run_pie_test_case("../p04015.py", input_content, expected_output)


def test_problem_p04015_1():
    input_content = "4 8\n7 9 8 9"
    expected_output = "5"
    run_pie_test_case("../p04015.py", input_content, expected_output)


def test_problem_p04015_2():
    input_content = "3 8\n6 6 9"
    expected_output = "0"
    run_pie_test_case("../p04015.py", input_content, expected_output)


def test_problem_p04015_3():
    input_content = "33 3\n3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3"
    expected_output = "8589934591"
    run_pie_test_case("../p04015.py", input_content, expected_output)


def test_problem_p04015_4():
    input_content = "8 5\n3 6 2 8 7 6 5 9"
    expected_output = "19"
    run_pie_test_case("../p04015.py", input_content, expected_output)
