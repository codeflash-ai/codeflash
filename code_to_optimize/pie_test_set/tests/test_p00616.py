from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00616_0():
    input_content = "4 3\nxy 4 4\nxz 1 2\nyz 2 3\n4 5\nxy 1 1\nxy 3 3\nxz 3 3\nyz 2 1\nyz 3 3\n0 0"
    expected_output = "52\n46"
    run_pie_test_case("../p00616.py", input_content, expected_output)


def test_problem_p00616_1():
    input_content = "4 3\nxy 4 4\nxz 1 2\nyz 2 3\n4 5\nxy 1 1\nxy 3 3\nxz 3 3\nyz 2 1\nyz 3 3\n0 0"
    expected_output = "52\n46"
    run_pie_test_case("../p00616.py", input_content, expected_output)
