from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02308_0():
    input_content = "2 1 1\n2\n0 1 4 1\n3 0 3 3"
    expected_output = (
        "1.00000000 1.00000000 3.00000000 1.00000000\n3.00000000 1.00000000 3.00000000 1.00000000"
    )
    run_pie_test_case("../p02308.py", input_content, expected_output)


def test_problem_p02308_1():
    input_content = "2 1 1\n2\n0 1 4 1\n3 0 3 3"
    expected_output = (
        "1.00000000 1.00000000 3.00000000 1.00000000\n3.00000000 1.00000000 3.00000000 1.00000000"
    )
    run_pie_test_case("../p02308.py", input_content, expected_output)
