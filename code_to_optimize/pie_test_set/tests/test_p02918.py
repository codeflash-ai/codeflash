from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02918_0():
    input_content = "6 1\nLRLRRL"
    expected_output = "3"
    run_pie_test_case("../p02918.py", input_content, expected_output)


def test_problem_p02918_1():
    input_content = "6 1\nLRLRRL"
    expected_output = "3"
    run_pie_test_case("../p02918.py", input_content, expected_output)


def test_problem_p02918_2():
    input_content = "9 2\nRRRLRLRLL"
    expected_output = "7"
    run_pie_test_case("../p02918.py", input_content, expected_output)


def test_problem_p02918_3():
    input_content = "13 3\nLRRLRLRRLRLLR"
    expected_output = "9"
    run_pie_test_case("../p02918.py", input_content, expected_output)


def test_problem_p02918_4():
    input_content = "10 1\nLLLLLRRRRR"
    expected_output = "9"
    run_pie_test_case("../p02918.py", input_content, expected_output)
