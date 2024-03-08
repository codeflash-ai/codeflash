from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00092_0():
    input_content = "10\n...*....**\n..........\n**....**..\n........*.\n..*.......\n..........\n.*........\n..........\n....*..***\n.*....*...\n10\n****.*****\n*..*.*....\n****.*....\n*....*....\n*....*****\n..........\n****.*****\n*..*...*..\n****...*..\n*..*...*..\n0"
    expected_output = "5\n3"
    run_pie_test_case("../p00092.py", input_content, expected_output)


def test_problem_p00092_1():
    input_content = "10\n...*....**\n..........\n**....**..\n........*.\n..*.......\n..........\n.*........\n..........\n....*..***\n.*....*...\n10\n****.*****\n*..*.*....\n****.*....\n*....*....\n*....*****\n..........\n****.*****\n*..*...*..\n****...*..\n*..*...*..\n0"
    expected_output = "5\n3"
    run_pie_test_case("../p00092.py", input_content, expected_output)
