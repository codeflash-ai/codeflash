from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03027_0():
    input_content = "2\n7 2 4\n12345 67890 2019"
    expected_output = "9009\n916936"
    run_pie_test_case("../p03027.py", input_content, expected_output)


def test_problem_p03027_1():
    input_content = "2\n7 2 4\n12345 67890 2019"
    expected_output = "9009\n916936"
    run_pie_test_case("../p03027.py", input_content, expected_output)
