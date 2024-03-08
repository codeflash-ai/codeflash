from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00425_0():
    input_content = "5\nNorth\nNorth\nEast\nSouth\nWest\n8\nWest\nNorth\nLeft\nSouth\nRight\nNorth\nLeft\nEast\n0"
    expected_output = "21\n34"
    run_pie_test_case("../p00425.py", input_content, expected_output)


def test_problem_p00425_1():
    input_content = "5\nNorth\nNorth\nEast\nSouth\nWest\n8\nWest\nNorth\nLeft\nSouth\nRight\nNorth\nLeft\nEast\n0"
    expected_output = "21\n34"
    run_pie_test_case("../p00425.py", input_content, expected_output)
