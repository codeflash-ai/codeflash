from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00777_0():
    input_content = (
        "4\n1 2 3\n10 20 30\n10\n1 2 2 1 5 5 1 8 8\n10 1 1 20 1 1 30 1 1\n3\n1 1\n1 1\n0"
    )
    expected_output = "80\n136\n2"
    run_pie_test_case("../p00777.py", input_content, expected_output)


def test_problem_p00777_1():
    input_content = (
        "4\n1 2 3\n10 20 30\n10\n1 2 2 1 5 5 1 8 8\n10 1 1 20 1 1 30 1 1\n3\n1 1\n1 1\n0"
    )
    expected_output = "80\n136\n2"
    run_pie_test_case("../p00777.py", input_content, expected_output)
