from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p04019_0():
    input_content = "SENW"
    expected_output = "Yes"
    run_pie_test_case("../p04019.py", input_content, expected_output)


def test_problem_p04019_1():
    input_content = "NSNNSNSN"
    expected_output = "Yes"
    run_pie_test_case("../p04019.py", input_content, expected_output)


def test_problem_p04019_2():
    input_content = "NNEW"
    expected_output = "No"
    run_pie_test_case("../p04019.py", input_content, expected_output)


def test_problem_p04019_3():
    input_content = "SENW"
    expected_output = "Yes"
    run_pie_test_case("../p04019.py", input_content, expected_output)


def test_problem_p04019_4():
    input_content = "W"
    expected_output = "No"
    run_pie_test_case("../p04019.py", input_content, expected_output)
