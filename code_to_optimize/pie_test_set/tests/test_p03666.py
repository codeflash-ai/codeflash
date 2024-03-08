from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03666_0():
    input_content = "5 1 5 2 4"
    expected_output = "YES"
    run_pie_test_case("../p03666.py", input_content, expected_output)


def test_problem_p03666_1():
    input_content = "48792 105960835 681218449 90629745 90632170"
    expected_output = "NO"
    run_pie_test_case("../p03666.py", input_content, expected_output)


def test_problem_p03666_2():
    input_content = "491995 412925347 825318103 59999126 59999339"
    expected_output = "YES"
    run_pie_test_case("../p03666.py", input_content, expected_output)


def test_problem_p03666_3():
    input_content = "5 1 5 2 4"
    expected_output = "YES"
    run_pie_test_case("../p03666.py", input_content, expected_output)


def test_problem_p03666_4():
    input_content = "4 7 6 4 5"
    expected_output = "NO"
    run_pie_test_case("../p03666.py", input_content, expected_output)
