from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02692_0():
    input_content = "2 1 3 0\nAB\nAC"
    expected_output = "Yes\nA\nC"
    run_pie_test_case("../p02692.py", input_content, expected_output)


def test_problem_p02692_1():
    input_content = "2 1 3 0\nAB\nAC"
    expected_output = "Yes\nA\nC"
    run_pie_test_case("../p02692.py", input_content, expected_output)


def test_problem_p02692_2():
    input_content = "1 0 9 0\nAC"
    expected_output = "No"
    run_pie_test_case("../p02692.py", input_content, expected_output)


def test_problem_p02692_3():
    input_content = "8 6 9 1\nAC\nBC\nAB\nBC\nAC\nBC\nAB\nAB"
    expected_output = "Yes\nC\nB\nB\nC\nC\nB\nA\nA"
    run_pie_test_case("../p02692.py", input_content, expected_output)


def test_problem_p02692_4():
    input_content = "3 1 0 0\nAB\nBC\nAB"
    expected_output = "No"
    run_pie_test_case("../p02692.py", input_content, expected_output)
