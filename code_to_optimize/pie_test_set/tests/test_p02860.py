from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02860_0():
    input_content = "6\nabcabc"
    expected_output = "Yes"
    run_pie_test_case("../p02860.py", input_content, expected_output)


def test_problem_p02860_1():
    input_content = "6\nabcadc"
    expected_output = "No"
    run_pie_test_case("../p02860.py", input_content, expected_output)


def test_problem_p02860_2():
    input_content = "6\nabcabc"
    expected_output = "Yes"
    run_pie_test_case("../p02860.py", input_content, expected_output)


def test_problem_p02860_3():
    input_content = "1\nz"
    expected_output = "No"
    run_pie_test_case("../p02860.py", input_content, expected_output)
