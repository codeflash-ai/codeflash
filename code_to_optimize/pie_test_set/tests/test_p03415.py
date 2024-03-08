from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03415_0():
    input_content = "ant\nobe\nrec"
    expected_output = "abc"
    run_pie_test_case("../p03415.py", input_content, expected_output)


def test_problem_p03415_1():
    input_content = "ant\nobe\nrec"
    expected_output = "abc"
    run_pie_test_case("../p03415.py", input_content, expected_output)


def test_problem_p03415_2():
    input_content = "edu\ncat\nion"
    expected_output = "ean"
    run_pie_test_case("../p03415.py", input_content, expected_output)
