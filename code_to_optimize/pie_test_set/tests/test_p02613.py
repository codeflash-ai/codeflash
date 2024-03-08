from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02613_0():
    input_content = "6\nAC\nTLE\nAC\nAC\nWA\nTLE"
    expected_output = "AC x 3\nWA x 1\nTLE x 2\nRE x 0"
    run_pie_test_case("../p02613.py", input_content, expected_output)


def test_problem_p02613_1():
    input_content = "6\nAC\nTLE\nAC\nAC\nWA\nTLE"
    expected_output = "AC x 3\nWA x 1\nTLE x 2\nRE x 0"
    run_pie_test_case("../p02613.py", input_content, expected_output)


def test_problem_p02613_2():
    input_content = "10\nAC\nAC\nAC\nAC\nAC\nAC\nAC\nAC\nAC\nAC"
    expected_output = "AC x 10\nWA x 0\nTLE x 0\nRE x 0"
    run_pie_test_case("../p02613.py", input_content, expected_output)
