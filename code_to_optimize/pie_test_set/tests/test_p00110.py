from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00110_0():
    input_content = "123+4X6=X79\n12X+4X6=X79\nXX22+89=X2XX"
    expected_output = "5\nNA\n1"
    run_pie_test_case("../p00110.py", input_content, expected_output)


def test_problem_p00110_1():
    input_content = "123+4X6=X79\n12X+4X6=X79\nXX22+89=X2XX"
    expected_output = "5\nNA\n1"
    run_pie_test_case("../p00110.py", input_content, expected_output)
