from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00263_0():
    input_content = (
        "8\n00000000\n80000000\n00000080\n00000040\n000000c0\n00000100\n80000780\n80000f70"
    )
    expected_output = "0.0\n-0.0\n1.0\n0.5\n1.5\n2.0\n-15.0\n-30.875"
    run_pie_test_case("../p00263.py", input_content, expected_output)


def test_problem_p00263_1():
    input_content = (
        "8\n00000000\n80000000\n00000080\n00000040\n000000c0\n00000100\n80000780\n80000f70"
    )
    expected_output = "0.0\n-0.0\n1.0\n0.5\n1.5\n2.0\n-15.0\n-30.875"
    run_pie_test_case("../p00263.py", input_content, expected_output)
