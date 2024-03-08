from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00042_0():
    input_content = (
        "50\n5\n60,10\n100,20\n120,30\n210,45\n10,4\n50\n5\n60,10\n100,20\n120,30\n210,45\n10,4\n0"
    )
    expected_output = "Case 1:\n220\n49\nCase 2:\n220\n49"
    run_pie_test_case("../p00042.py", input_content, expected_output)


def test_problem_p00042_1():
    input_content = (
        "50\n5\n60,10\n100,20\n120,30\n210,45\n10,4\n50\n5\n60,10\n100,20\n120,30\n210,45\n10,4\n0"
    )
    expected_output = "Case 1:\n220\n49\nCase 2:\n220\n49"
    run_pie_test_case("../p00042.py", input_content, expected_output)
