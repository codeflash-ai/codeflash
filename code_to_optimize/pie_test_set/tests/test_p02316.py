from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02316_0():
    input_content = "4 8\n4 2\n5 2\n2 1\n8 3"
    expected_output = "21"
    run_pie_test_case("../p02316.py", input_content, expected_output)
