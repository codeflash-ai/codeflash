from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00083_0():
    input_content = "2005 9 3\n1868 12 2\n1868 9 7"
    expected_output = "heisei 17 9 3\nmeiji 1 12 2\npre-meiji"
    run_pie_test_case("../p00083.py", input_content, expected_output)


def test_problem_p00083_1():
    input_content = "2005 9 3\n1868 12 2\n1868 9 7"
    expected_output = "heisei 17 9 3\nmeiji 1 12 2\npre-meiji"
    run_pie_test_case("../p00083.py", input_content, expected_output)
