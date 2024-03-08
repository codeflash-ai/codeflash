from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00271_0():
    input_content = "30 19\n39 20\n19 18\n25 20\n22 21\n23 10\n10 -10"
    expected_output = "11\n19\n1\n5\n1\n13\n20"
    run_pie_test_case("../p00271.py", input_content, expected_output)


def test_problem_p00271_1():
    input_content = "30 19\n39 20\n19 18\n25 20\n22 21\n23 10\n10 -10"
    expected_output = "11\n19\n1\n5\n1\n13\n20"
    run_pie_test_case("../p00271.py", input_content, expected_output)
