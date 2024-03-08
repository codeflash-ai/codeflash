from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00537_0():
    input_content = "4 4\n1 3 2 4\n120 90 100\n110 50 80\n250 70 130"
    expected_output = "550"
    run_pie_test_case("../p00537.py", input_content, expected_output)


def test_problem_p00537_1():
    input_content = "4 4\n1 3 2 4\n120 90 100\n110 50 80\n250 70 130"
    expected_output = "550"
    run_pie_test_case("../p00537.py", input_content, expected_output)
