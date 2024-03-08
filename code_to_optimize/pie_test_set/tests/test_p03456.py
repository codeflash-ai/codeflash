from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03456_0():
    input_content = "1 21"
    expected_output = "Yes"
    run_pie_test_case("../p03456.py", input_content, expected_output)


def test_problem_p03456_1():
    input_content = "1 21"
    expected_output = "Yes"
    run_pie_test_case("../p03456.py", input_content, expected_output)


def test_problem_p03456_2():
    input_content = "12 10"
    expected_output = "No"
    run_pie_test_case("../p03456.py", input_content, expected_output)


def test_problem_p03456_3():
    input_content = "100 100"
    expected_output = "No"
    run_pie_test_case("../p03456.py", input_content, expected_output)
