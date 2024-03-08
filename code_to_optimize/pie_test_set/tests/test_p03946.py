from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03946_0():
    input_content = "3 2\n100 50 200"
    expected_output = "1"
    run_pie_test_case("../p03946.py", input_content, expected_output)
