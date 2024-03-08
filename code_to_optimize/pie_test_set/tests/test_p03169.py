from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03169_0():
    input_content = "3\n1 1 1"
    expected_output = "5.5"
    run_pie_test_case("../p03169.py", input_content, expected_output)


def test_problem_p03169_1():
    input_content = "1\n3"
    expected_output = "3"
    run_pie_test_case("../p03169.py", input_content, expected_output)


def test_problem_p03169_2():
    input_content = "2\n1 2"
    expected_output = "4.5"
    run_pie_test_case("../p03169.py", input_content, expected_output)


def test_problem_p03169_3():
    input_content = "3\n1 1 1"
    expected_output = "5.5"
    run_pie_test_case("../p03169.py", input_content, expected_output)


def test_problem_p03169_4():
    input_content = "10\n1 3 2 3 3 2 3 2 1 3"
    expected_output = "54.48064457488221"
    run_pie_test_case("../p03169.py", input_content, expected_output)
