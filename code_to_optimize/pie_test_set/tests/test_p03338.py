from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03338_0():
    input_content = "6\naabbca"
    expected_output = "2"
    run_pie_test_case("../p03338.py", input_content, expected_output)


def test_problem_p03338_1():
    input_content = "10\naaaaaaaaaa"
    expected_output = "1"
    run_pie_test_case("../p03338.py", input_content, expected_output)


def test_problem_p03338_2():
    input_content = "6\naabbca"
    expected_output = "2"
    run_pie_test_case("../p03338.py", input_content, expected_output)


def test_problem_p03338_3():
    input_content = "45\ntgxgdqkyjzhyputjjtllptdfxocrylqfqjynmfbfucbir"
    expected_output = "9"
    run_pie_test_case("../p03338.py", input_content, expected_output)
