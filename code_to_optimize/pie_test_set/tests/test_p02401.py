from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02401_0():
    input_content = "1 + 2\n56 - 18\n13 * 2\n100 / 10\n27 + 81\n0 ? 0"
    expected_output = "3\n38\n26\n10\n108"
    run_pie_test_case("../p02401.py", input_content, expected_output)


def test_problem_p02401_1():
    input_content = "1 + 2\n56 - 18\n13 * 2\n100 / 10\n27 + 81\n0 ? 0"
    expected_output = "3\n38\n26\n10\n108"
    run_pie_test_case("../p02401.py", input_content, expected_output)
