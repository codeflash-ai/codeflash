from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03704_0():
    input_content = "63"
    expected_output = "2"
    run_pie_test_case("../p03704.py", input_content, expected_output)


def test_problem_p03704_1():
    input_content = "63"
    expected_output = "2"
    run_pie_test_case("../p03704.py", input_content, expected_output)


def test_problem_p03704_2():
    input_content = "864197532"
    expected_output = "1920"
    run_pie_test_case("../p03704.py", input_content, expected_output)


def test_problem_p03704_3():
    input_content = "75"
    expected_output = "0"
    run_pie_test_case("../p03704.py", input_content, expected_output)
