from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03836_0():
    input_content = "0 0 1 2"
    expected_output = "UURDDLLUUURRDRDDDLLU"
    run_pie_test_case("../p03836.py", input_content, expected_output)


def test_problem_p03836_1():
    input_content = "0 0 1 2"
    expected_output = "UURDDLLUUURRDRDDDLLU"
    run_pie_test_case("../p03836.py", input_content, expected_output)


def test_problem_p03836_2():
    input_content = "-2 -2 1 1"
    expected_output = "UURRURRDDDLLDLLULUUURRURRDDDLLDL"
    run_pie_test_case("../p03836.py", input_content, expected_output)
