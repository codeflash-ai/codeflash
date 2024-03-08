from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03075_0():
    input_content = "1\n2\n4\n8\n9\n15"
    expected_output = "Yay!"
    run_pie_test_case("../p03075.py", input_content, expected_output)


def test_problem_p03075_1():
    input_content = "1\n2\n4\n8\n9\n15"
    expected_output = "Yay!"
    run_pie_test_case("../p03075.py", input_content, expected_output)


def test_problem_p03075_2():
    input_content = "15\n18\n26\n35\n36\n18"
    expected_output = ":("
    run_pie_test_case("../p03075.py", input_content, expected_output)
