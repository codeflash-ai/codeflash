from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02242_0():
    input_content = (
        "5\n0 3 2 3 3 1 1 2\n1 2 0 2 3 4\n2 3 0 3 3 1 4 1\n3 4 2 1 0 1 1 4 4 3\n4 2 2 1 3 3"
    )
    expected_output = "0 0\n1 2\n2 2\n3 1\n4 3"
    run_pie_test_case("../p02242.py", input_content, expected_output)
