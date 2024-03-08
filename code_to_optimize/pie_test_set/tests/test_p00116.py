from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p00116_0():
    input_content = "10 10\n...*....**\n..........\n**....**..\n........*.\n..*.......\n**........\n.*........\n..........\n....*..***\n.*....*...\n10 10\n..*....*..\n.*.*...*..\n*****..*..\n*...*..*..\n*...*..*..\n..........\n****.*...*\n..*..*...*\n.*...*...*\n****..***.\n2 3\n...\n...\n0 0"
    expected_output = "28\n12\n6"
    run_pie_test_case("../p00116.py", input_content, expected_output)


def test_problem_p00116_1():
    input_content = "10 10\n...*....**\n..........\n**....**..\n........*.\n..*.......\n**........\n.*........\n..........\n....*..***\n.*....*...\n10 10\n..*....*..\n.*.*...*..\n*****..*..\n*...*..*..\n*...*..*..\n..........\n****.*...*\n..*..*...*\n.*...*...*\n****..***.\n2 3\n...\n...\n0 0"
    expected_output = "28\n12\n6"
    run_pie_test_case("../p00116.py", input_content, expected_output)
